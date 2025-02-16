from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size = 50257      # no. of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    block_size = 1024       # no. of positional indices / max seq len
    n_embd = 768            # embedding dimension
    n_layer = 12
    n_head = 12


class GPT(nn.Module):
    """
    Implements GPT2 model from transformers, so that we can load weights from hf
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # gpt model state dict
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            h = nn.ModuleList([
                TransformerBlock(config)   # transformer block
                for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # classification head

    def forward(self, idx, targets=None):
        # idx is token_ids : (B, T), C = n_emb
        B, T = idx.shape  # batch_size, seq_len
        # Note: Transformer architecture has no constraints on max seq len. It can support seq len of size infinite theortically.
        # There are only computational contrains with KV cache as the size grows exponentially with seq len.
        # However in GPT2 case, the positional encoder is fixed sized and thus we can't allow longer sequences.
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # position and token embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T) just an array from 0-T
        pos_emb = self.transformer.wpe(pos)     # (B, T, C)
        tok_emb = self.transformer.wte(idx)     # (B, T, C)
        x = tok_emb + pos_emb
        # forward transformer blocks
        for block in self.transformer.h:
            x = block(x)                        # (B, T, C)
        # forward final layer norm and classifier
        x = self.transformer.ln_f(x)            # (B, T, C)
        logits = self.lm_head(x)                # (B, T, vocab_size)
        
        # calculate loss for training
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size) - input to ce is logits
                targets.view(-1)                    # (B*T) - target one dimensional target ids
            )
        
        return logits, loss



class TransformerBlock(nn.Module):
    """
    Implements GPT transformer block
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)     # layer norm 1
        self.attn = CausalSelfAttention(config)     # attention module
        self.ln_2 = nn.LayerNorm(config.n_embd)     # layer norm 1
        self.mlp = MLP(config)                      # fully connected layer

    def forward(self, x):
        # layer norm at begining of submodules 
        # and with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """
    Implements Multi-Head Attention block
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections as single matrix operation
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)     
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)         # linear proj on attention output
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        # query, key, value projections - nh: n_heads, hs: C/nh
        qkv = self.c_attn(x)                                            # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=-1)                        # (B, T, C) , ..
        # split heads
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)    # (B, nh, T, hs) # transpose n_head with seq_len for attention calculation
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)     # flash attention # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side   # (B, T, C)
        # output projection
        y = self.c_proj(y)  # (B, T, C)
        return y


class MLP(nn.Module):
    """
    Implements the MLP module
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        fc_dim = config.n_embd * 4
        self.c_fc = nn.Linear(config.n_embd, fc_dim)     # fan out
        self.gelu = nn.GELU(approximate='tanh')          # Gelu non-linearlity
        self.c_proj = nn.Linear(fc_dim, config.n_embd)   # fan in

    def forward(self, x):
        # x: (B, T, C)
        x = self.c_fc(x)        # (B, T, fc)
        x = self.gelu(x)        # (B, T, fc)
        x = self.c_proj(x)      # (B, T, C)
        return x
