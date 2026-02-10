import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
import tiktoken

from model import GPTConfig, GPT, GPTConfigSmall
from data_loader import TextDataset

out_dir = 'out'
eval_interval = 250
eval_iters = 200

if __name__ == '__main__':
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        from torch.distributed import init_process_group, destroy_process_group
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

    gpt_config = GPTConfigSmall()

    B = 16
    T = gpt_config.block_size

    total_batch_size = 64*T # 2**19, ~0.5M, in number of tokens

    print(f"B: {B}, T: {T}, ddp_world_size: {ddp_world_size}, total_batch_size: {total_batch_size}")

    assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"

    grad_accum_steps = total_batch_size // (B*T*ddp_world_size)

    if master_process:
        print(f"Training {total_batch_size} tokens")
        print(f"Training with {B} tokens per batch and {T} sequence length")
        print(f"Training with {grad_accum_steps} gradient accumulation steps")

    # Dataset and DataLoader setup
    train_dataset = TextDataset('input.txt', T, split='train')
    val_dataset = TextDataset('input.txt', T, split='val')

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=False, pin_memory=True, persistent_workers=True, num_workers=12)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    def get_batch(split):
        if split == 'train':
            x, y = next(train_iter)
        else:
            x, y = next(val_iter)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    @torch.no_grad()
    def estimate_loss():
        """Estimate mean train/val loss over a few batches, like the reference script."""
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    torch.set_float32_matmul_precision('high')

    model = GPT(gpt_config)

    model.to(device)

    use_compile = False
    if use_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 100
    max_steps = 2000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

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

        return min_lr + coeff * ( max_lr - min_lr)
        
    optimizer = raw_model.configure_optimizers(weight_decay=0.2, learning_rate=max_lr, device_type=device_type)


    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    if master_process:
        # Use UTF-8 so we can safely write any unicode characters (e.g. Japanese) on Windows
        with open(log_file, "w", encoding="utf-8") as f:
            pass

    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step % eval_interval == 0 or last_step:
            # evaluate losses and generate a sample, similar to the reference script
            if master_process:
                losses = estimate_loss()
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")
            # switch to eval mode and disable grad for generation
            model.eval()
            with torch.no_grad():
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                out = enc.decode(model.generate(context, max_new_tokens=T)[0].tolist())
                print(out)
                # Write generated text using UTF-8 to avoid UnicodeEncodeError on Windows cp1252
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(out + "\n")
                
            model.train()
            print(f"step {step} of {max_steps} took {time.time() - t0:.2f}s")

        # Train for one step

        model.train()
        optimizer.zero_grad()

        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = get_batch('train')
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True) # non_blocking=True is faster  for async transfer

            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # still work for and faster than no_sync() context manager

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            # Andrej notes
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

        token_processed = B * T * ddp_world_size * grad_accum_steps
        tokens_pre_sec = token_processed / (dt + 1e-5)

        if master_process:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"step {step} loss {loss_accum.item():.4f} lr {lr:.6f} time {dt:.2f}s tokens/sec {tokens_pre_sec:.2f}\ lr {lr:.6f}\n")
            print(f"step {step} loss {loss_accum.item():.4f} lr {lr:.6f} time {dt:.2f}s tokens/sec {tokens_pre_sec:.2f}\ lr {lr:.6f}")

    # final generation after training
    if master_process:
        model.eval()
        with torch.no_grad():
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(enc.decode(model.generate(context, max_new_tokens=T)[0].tolist()))
        model.train()
    if ddp:
        destroy_process_group()
