import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import tiktoken

from model import GPTConfig, GPT

out_dir = 'out'
eval_interval = 250


ddp = int(os.environ.get("RANK", -1)) != -1

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get(['RANK']))
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    print(f"Running on device: {device}")
    print(f"DDP_rank: {ddp_rank}, DDP_local_rank: {ddp_local_rank}, DDP_world_size: {ddp_world_size}")
    

else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")
total_batch_size = 524277 # 2**19, ~0.5M, in number of tokens

B = 64
T = 1024

assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

grad_accum_steps = total_batch_size // (B*T*ddp_world_size)

if master_process:
    print(f"Training {total_batch_size} tokens")
    print(f"Training with {B} tokens per batch and {T} sequence length")
    print(f"Training with {grad_accum_steps} gradient accumulation steps")

train_loader = None
val_loader = None

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig(block_size=50304))

model.to(device)

use_compile = True
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

def get_cosine_lr(it):
    #warmup
    if it < warmup_steps:
        return max_lr * it / warmup_steps

    #if after max_steps 

    if it > max_steps:
        return min_lr

    # in between
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff ( max_lr - min_lr)
    
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)


log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
if master_process:
    with open(log_file, "w") as f:
        pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    if step % eval_interval == 0 or last_step:
        #TODO: evaluate
        print(f"step {step} of {max_steps} took {time.time() - to:.2f}s")

    # Train for one step

    model.train()
    optimizer.zero_grad()

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # non_blocking=True is faster  for async transfer

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # still work for and faster than no_sync() context manager

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Andrew notes
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right

        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # we do this since we need just the value, not the computation graph

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # with reuce() we can even set to which device to reduce to

    lr = get_cosine_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #TODO better to use optimizer.step()

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: get_lr(step))
    # )

    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()
    
    t1 = time.time()
    dt = t1 - t0

    token_processed = train_loader.B * train_loader.T * ddp_world_size * grad_accum_steps
    tokens_pre_sec = token_processed / dt

    if master_process:
        with open(log_file, "a") as f:
            f.write(f"step {step} loss {loss_accum.item():.4f} lr {lr:.6f} time {dt:.2f}s tokens/sec {tokens_pre_sec:.2f}\n")
        print(f"step {step} loss {loss_accum.item():.4f} lr {lr:.6f} time {dt:.2f}s tokens/sec {tokens_pre_sec:.2f}")

    
if ddp:
    destroy_process_group()





    


    


    
