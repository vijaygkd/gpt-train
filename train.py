import os
import time
import math
import torch

from gpt2 import GPT, GPTConfig, DataLoaderLite


# -----------------------------------------------------------------------------
# simple launch: 1 GPU or CPU or MPS
# python train.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])                  # rank of current GPU
    ddp_local_rank = int(os.environ['LOCAL_RANK'])      # local rank in multi node
    ddp_world_size = int(os.environ['WORLD_SIZE'])      # total no. of GPU or processes running
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

# IMP - because of the seed, identical copies of model are created in each GPU process
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


## -------------------------------------------------
## DATASET
# batch size - 0.5M as per GPT paper
total_batch_size = 524288    # 2^19  ~0.5M
B = 16      # micro batch size
T = 1024    # seq length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total batch size: {total_batch_size} tokens")
    print(f"=> grad accum steps: {grad_accum_steps}")

# print("I am GPU: ", ddp_rank)
# import sys; sys.exit(0)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

## -------------------------------------------------
## MODEL
print(f"Device: {device}")
nice_vocab_size = 50304     # nice power of 2
model = GPT(GPTConfig(vocab_size=nice_vocab_size))
model.to(device)
# torch compile to optimize
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

## -------------------------------------------------
## OPTIMIZER
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715      # 375M token warmup / total_batch_size - as per GPT3 paper
max_steps = 19073     # 10B (dataset) / (total_batch_size) 
def get_lr(it):
    # 1) warm up phase
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) min learning rate after max_steps
    if it > max_steps:
        return min_lr
    # 3) consine decay to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)  # goes from 0 to 1
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff goes from 1 to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer hyperparams based on GPT-3 paper
# optimizer = torch.optim.AdamW(
#     model.parameters(), 
#     lr=3e-4,
#     betas=(0.9, 0.95),
#     eps=1e-8
# )

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    device_type=device_type
)


## -------------------------------------------------
## TRAIN

# set computation to TF32 instead of F32
torch.set_float32_matmul_precision('high')

# train loop
for step in range(max_steps):
    t0 = time.time()

    # Evaluate on Validation loss dataset once in a while
    if step % 100 == 0: # or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            # with open(log_file, "a") as f:
            #     f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # if step > 0 and (step % 5000 == 0 or last_step):
            #     # optionally write model checkpoints
            #     checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            #     checkpoint = {
            #         'model': raw_model.state_dict(),
            #         'config': raw_model.config,
            #         'step': step,
            #         'val_loss': val_loss_accum.item()
            #     }
            #     # you might also want to add optimizer.state_dict() and
            #     # rng seeds etc., if you wanted to more exactly resume training
            #     torch.save(checkpoint, checkpoint_path)

    # Training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        # data
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # BF16 mixed precision training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # scale the loss to account for gradient accumulation
        # loss.backwards() keeps adding losses during successive steps
        # normalize it so we have mean loss during gradient step
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # sync avg of grads between processes on last grad accum step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        # backprop
        loss.backward()
    if ddp:
        # calculate avg loss_accum across process and distribute
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # gradient cliping - as per GPT3 paper
    # to avoid shocking the model during training due to big loss in a batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # instrument
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process: print(f"step {step} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()