import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

out_dir = 'out'
eval_interval = 200


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
    
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally

    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //=ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Training in {out_dir}...")

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
