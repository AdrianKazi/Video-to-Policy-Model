import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout
        self.QKV = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.W0 = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        B, T, E = x.shape
        qkv = self.QKV(x)
        q, k, v = torch.split(qkv, E, dim=2)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout_p,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        return self.W0(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.layernorm_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.mlp_1 = nn.Linear(embed_dim, 4 * embed_dim, bias=True)
        self.gelu = nn.GELU()
        self.mlp_2 = nn.Linear(4 * embed_dim, embed_dim, bias=True)
        self.trn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_att = self.layernorm_1(x)
        x_att = x + self.trn_dropout(self.attn(x_att))
        x_ff = self.layernorm_2(x_att)
        x_ff = self.mlp_2(self.gelu(self.mlp_1(x_ff)))

        return x_att + self.trn_dropout(x_ff)


class ActionEmbeddingModel(nn.Module):
    def __init__(self, flattened_dim, seq_len, embed_dim=256, n_heads=4, n_blocks=4, dropout=0.1):
        super().__init__()
        self.flattened_dim = flattened_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(flattened_dim, embed_dim, bias=True)
        self.wpe = nn.Embedding(seq_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.layernorm_final = nn.LayerNorm(embed_dim, eps=1e-5)
        self.final_head = nn.Linear(embed_dim, flattened_dim, bias=True)
        self.apply(self.weight_inits)

    def weight_inits(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def forward(self, z):
        token_emb = self.input_proj(z)
        posit_emb = self.wpe(torch.arange(z.shape[1], device=z.device))
        x = token_emb + posit_emb
        x = self.emb_dropout(x)
        x = self.transformer_blocks(x)
        x = self.layernorm_final(x)

        return self.final_head(x)

