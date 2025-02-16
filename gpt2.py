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


# ----------------------------------------------------------------------#
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('mps')

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to('mps')

torch.manual_seed(42)
# generate!
while x.size(1) < max_length: # max_length=30
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x)[0] # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(5):
    tokens = x[i, :30].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)