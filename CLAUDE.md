# CLAUDE.md - myGPT Project Guide

## Project Overview
Minimal GPT implementation in PyTorch inspired by Andrej Karpathy's nanoGPT. Decoder-only transformer with causal self-attention, supporting training from scratch or fine-tuning from pretrained GPT-2 weights.

## Project Structure
- `model.py` — GPT model architecture (GPTConfig ~124M params, GPTConfigSmall ~10M params), attention, MLP, generation, weight loading from HuggingFace
- `train.py` — Training script for full GPT-2 124M model (DDP support, gradient accumulation, cosine LR)
- `train_small.py` — Training script for smaller ~10M model using DataLoader-based pipeline
- `data_loader.py` — `TextDataset` class: tokenizes text with tiktoken GPT-2 BPE, 90/10 train/val split, returns (input, target) pairs shifted by one token
- `input.txt` — Training data (plain text file)

## Commands
```bash
# Train small model (~10M params)
python train_small.py

# Train full model (~124M params)
python train.py

# Multi-GPU training with DDP
torchrun --standalone --nproc_per_node=4 train_small.py

# Run data_loader tests
python data_loader.py
```

## Dependencies
- Python 3.8+
- PyTorch 2.0+ (for Flash Attention via scaled_dot_product_attention)
- tiktoken
- transformers (for loading pretrained GPT-2 weights)
- numpy

## Architecture Notes
- Pre-norm transformer (LayerNorm before attention/MLP, not after)
- Weight tying between token embedding (`wte`) and output projection (`lm_head`)
- Flash Attention auto-detected; falls back to manual attention with causal mask
- `configure_optimizers()` splits params into decay (dim >= 2) and no-decay groups for AdamW
- Mixed precision via `torch.autocast` with bfloat16
- Residual projection weights (`c_proj`) get scaled init: `std = 0.02 / sqrt(2 * n_layers)`

## Key Configurations
| Config | Layers | Heads | Embed | Block Size | Params |
|--------|--------|-------|-------|------------|--------|
| GPTConfigSmall | 6 | 6 | 384 | 256 | ~10M |
| GPTConfig | 12 | 12 | 768 | 1024 | ~124M |

## Known Issues in train.py
- Line 25: `os.environ.get(['RANK'])` passes a list instead of a string — should be `os.environ.get('RANK')`
- Line 108: `coeff ( max_lr - min_lr)` is missing `*` operator — should be `coeff * (max_lr - min_lr)`
- Line 55: `total_batch_size = 524277` comment says 2**19 but 2**19 = 524288
- Lines 71-72: `train_loader` and `val_loader` are set to `None` but `train_loader.next_batch()` is called in the training loop (line 135) — data loading is not wired up

## Code Conventions
- Follows nanoGPT naming: `c_attn`, `c_proj`, `c_fc` for linear layers; `wte`, `wpe` for embeddings
- Config uses `n_embed`, `n_heads`, `n_layers`, `n_hidden` (not the HuggingFace naming)
- Logging goes to `log/log.txt` with UTF-8 encoding
- Training loop uses manual LR scheduling via `param_group['lr']` assignment (not a torch scheduler)
