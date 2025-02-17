import time
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

# optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# dataloader
from gpt2 import DataLoaderLite
train_loader = DataLoaderLite(B=16, T=1024)

# set computation to TF32 instead of F32
torch.set_float32_matmul_precision('high')

# train loop
for i in range(50):
    t0 = time.time()
    # ata
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # optim
    optimizer.zero_grad()
    # BF16 mixed precision training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    # step
    loss.backward()
    optimizer.step()

    # instrument
    if device == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {i} | loss: {loss.item()} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

