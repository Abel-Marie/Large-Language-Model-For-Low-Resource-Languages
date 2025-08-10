import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.q_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('causal_mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        B, T, C = x.shape

        keys = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        mask = self.causal_mask[:T, :T].bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2).reshape(B, T, self.d_out)
        return self.out_proj(context)
