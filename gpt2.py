import inspect
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257      # no. of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    block_size: int = 1024       # no. of positional indices / max seq len
    n_embd: int = 768            # embedding dimension
    n_layer: int = 12
    n_head: int = 12


class CausalSelfAttention(nn.Module):
    """
    Implements Multi-Head Attention block
    """
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Q, K, V projections as single matrix operation
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)     
        # layer output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)         # linear proj on attention output
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, n_embd
        # query, key, value projections - nh: n_heads, hs: C/nh
        qkv = self.c_attn(x)                                            # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=-1)                        # (B, T, C) , ..
        # split heads
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2)    # (B, nh, T, hs) # transpose n_head with seq_len for attention calculation
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2)    # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2)    # (B, nh, T, hs)
        
        # raw attention
        # att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, hs)

        # flash attention
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
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # x: (B, T, C)
        x = self.c_fc(x)        # (B, T, fc)
        x = self.gelu(x)        # (B, T, fc)
        x = self.c_proj(x)      # (B, T, C)
        return x


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
        # weight sharing scheme - token embedding and output head 
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)  # apply to all modules

    def _init_weights(self, module):
        # following GPT-2 init hyper-parameters
        if isinstance(module, nn.Linear):
            std = 0.02
            # scaling factor for linear layers by 1/sqrt(no_layers) 
            # this is to control residual stream stddev close to 1, otherwise it keeps growing
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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


    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # todo : move to training module
    def configure_optimizers(self, weight_decay, learning_rate, device_type, betas=(0.9, 0.95), eps=1e-8):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, fused=use_fused)
        return optimizer


# -----------------------------------------------------------------------------
import os
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite: 
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    # tiny shakespear init method
    # def __init__(self, B, T, process_rank, num_processes):
    #     self.B = B # batch size
    #     self.T = T # sequence length
    #     self.process_rank = process_rank
    #     self.num_processes = num_processes
    #     # load the input file
    #     with open('input.txt', 'r', encoding='utf-8') as f:
    #         text = f.read()
    #     # encode text to tokens using tiktoken GPT-2 encoder
    #     enc = tiktoken.get_encoding("gpt2")
    #     tokens = enc.encode(text)        
    #     # load first shard
    #     self.tokens = torch.tensor(tokens)
    #     # calculate and print stats
    #     n_tokens = len(self.tokens)
    #     n_batches = n_tokens // (self.B * self.T)
    #     print(f"Dataset stats:")
    #     print(f"- {n_tokens:,} tokens")
    #     print(f"- {n_batches:,} batches per epoch")
    #     # partition the data for each process as different start position
    #     self.current_position = self.B * self.T * self.num_processes


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T +1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# ----------------------------------------------------------------------#
