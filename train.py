import time
import math
import torch
from gpt2 import GPT, GPTConfig, DataLoaderLite


torch.manual_seed(1337)

device = 'cuda'


print(f"Device: {device}")
# nice vocab no.
nice_vocab_size = 50304     # nice power of 2
model = GPT(GPTConfig(vocab_size=nice_vocab_size))
model.to(device)
# torch compile
model = torch.compile(model)


## -------------------------------------------------
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
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

optimizer = model.configure_optimizers(
    weight_decay=0.1,
    lr=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    device_type=device
)

## -------------------------------------------------

# dataloader
from gpt2 import DataLoaderLite
train_loader = DataLoaderLite(B=16, T=1024)

# set computation to TF32 instead of F32
torch.set_float32_matmul_precision('high')

# train loop
for step in range(max_steps):
    t0 = time.time()
    # ata
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # optim
    optimizer.zero_grad()
    # BF16 mixed precision training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    # backprop
    loss.backward()
    # gradient cliping - as per GPT3 paper
    # to avoid shocking the model during training due to big loss in a batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    # instrument
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

