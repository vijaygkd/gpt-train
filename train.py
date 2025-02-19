import os
import time
import math

import torch
from torch.nn import functional as F
import tiktoken

from gpt2 import GPT, GPTConfig, DataLoaderLite
from hellaswag import render_example, iterate_examples, get_most_likely_row

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

print(f"Device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

# IMP - because of the seed, identical copies of model are created in each GPU process
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


## -------------------------------------------------
## DATASET
# batch size - 0.5M as per GPT paper
total_batch_size = 524288    # 2^19  ~0.5M
B = 64      # micro batch size
T = 1024    # seq length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total batch size: {total_batch_size} tokens")
    print(f"=> grad accum steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

## -------------------------------------------------
## MODEL
nice_vocab_size = 50304     # nice power of 2
model = GPT(GPTConfig(vocab_size=nice_vocab_size))
model.to(device)
# torch compile to optimize
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model
enc = tiktoken.get_encoding("gpt2")

## -------------------------------------------------
## OPTIMIZER
max_lr = 6e-4 * 3       # increase learning rate
min_lr = max_lr * 0.1
warmup_steps = 715      # 375M token warmup / total_batch_size - as per GPT3 paper
steps_per_epoch = 19073     # 10B (dataset) / (total_batch_size) 
epochs = 3              # train for 30B
max_steps = steps_per_epoch * epochs
print(f"Total steps: {max_steps} | Epochs: {epochs}")
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
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    device_type=device_type
)


## -------------------------------------------------
## TRAIN

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
log_interval = 250              # log every n steps
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# set computation to TF32 instead of F32
torch.set_float32_matmul_precision('high')

# Train Loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate on Validation loss dataset once in a while
    if step % log_interval == 0 or last_step:
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
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    # once in a while evaluate hellaswag
    if (step % log_interval == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # generate text from the model (except step 0, which is noise)
    if ((step > 0 and step % log_interval == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # Training Step
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        # data
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            # sync avg grad between processes on last grad accum step
            # so that all copies of model get same gradients and weights stay in sync
            model.require_backward_grad_sync = (micro_step == grad_accum_steps-1)
        # BF16 mixed precision training with autocast
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            # forward pass
            logits, loss = model(x, y)
        # scale the loss to account for gradient accumulation
        # loss.backwards() keeps adding losses during successive steps
        # normalize it so we have mean loss during gradient step
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # backprop
        loss.backward()

    if ddp:
        # calculate avg loss_accum across process and distribute to all processes 
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
    if master_process: 
        print(f"step {step} | loss: {loss_accum:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

    # Model weight Checkpointing
    if step > 0 and (step % log_interval == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_name=f"model_{step:05d}"
        checkpoint_path = os.path.join(log_dir, f"{checkpoint_name}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'val_loss': val_loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)

if ddp:
    destroy_process_group()