import inspect
import torch

import torch.nn as nn
import torch.nn.functional as F
import math

class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_embed: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Paranetere(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embed % config.n_heads == 0

        #key, query, value projections for all heads, but in a batch

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_heads
        self.n_embed = config.n_embed
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is still experimental

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention may significantly speed up training, but support is only available in PyTorch 2.0 or later.")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embed)

        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y



class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = F.gelu()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x= self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed), # fast table lookup embedding
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.dropout), # dropout, but preferrably we should use nowadays weight_decay
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.n_embed, bias=config.bias), # to use rms_norm instead of layer_norm

        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight #weight tying

        self.apply(self._init_weights)

        # apply special scaled initialization to the residual paths
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" %  (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodes_params = sum(p.numel() for p in nodecay_params)

        # if master_process:
        print(f"num decay params: {num_decay_params}, num nodes params: {num_nodes_params}")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # checks if the __init__ has a parameter called fused
        print(f"fused available: {fused_available}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=fused_available)

        return optimizer

    
    def forward(self, idx, targets=None):
        device = idx.device

        b,t = idx.size()

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch)

        tok_emb = self.transformer.wte(idx) #b, t, n_embed)
        pos_emb = self.transformer.wpe(pos) #b, t, n_embed)

        x = self.transformer.drop(tok_emb + pos_emb)

        if targets is not None:
            # train
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # ignore index is "if targets that has -1 would not be used for calcualte the loss"
        else:
            #inference

            logits = self.lm_head(x[:, [-1], :]) # x after x[:, [-1], :] is (b, 1, n_embed) and then we get finally (b, 1, vocab_size)

            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if sequence context is too long we crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits,_ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1) # B, vocab_size

            idx_next = torch.multinomial(probs, num_samples=1) # B, 1

            idx = torch.cat((idx,idx_next), dim=1) # B, t+1 

        return idx