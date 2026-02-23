import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils import init_weight_xavier, init_weight_zero, init_weight_norm
from ..base_model import BaseModel
from .. import register_model

@torch.compile
def modulate(x, shift, scale):
    return x * (1 + scale) + shift

@register_model("LDM")
class LDM(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.encoder_latent_dim, args.hidden_dim,
                   args.embed_dim, args.num_heads,
                   args.dropout, args.mlp_ratio,
                   args.max_step, args.num_block, args.batch_size,
                )

    def __init__(self, input_dim, hidden_dim, embed_dim, num_heads, dropout, mlp_ratio,
                 max_step, num_block, sample_rate):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.p_embed = PositionEmbedding(
            sample_rate, input_dim, hidden_dim)
        self.t_embed = DiffusionEmbedding(
            max_step, embed_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

    def encode(self, x, t):
        x = self.p_embed(x)
        t = self.t_embed(t)
        for block in self.blocks:
            x = block(x, t)
        
        return x, t
    
    def decode(self, x, t):
        x = self.final_layer(x, t)
        return x
    
    def forward(self, x, t):
        x, t_emb = self.encode(x, t)
        x = self.decode(x, t_emb)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x): 
        x = self.projection(x)
        return x + self.embedding[:x.size(0), :]

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim // 2).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / (hidden_dim // 2))     # [P,E]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1) 
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = MultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

    def forward(self, x, t):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(mod_x, mod_x, mod_x).squeeze(0)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)).squeeze(0)
        return x

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)

    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        return self.projection(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim // 2).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / (embed_dim // 2))  # [T, E]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class DotProductAttention(nn.Module):
    """
    Query shape: [batch_size, query_num, query_key_dim]
    Key shape: [batch_size, key_value_num, query_key_dim]
    Value shape: [batch_size, key_value_num, value_dim]
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        query_key_dim = queries.shape[-1]
        attention_weights = F.softmax(
            torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(query_key_dim),
            dim=-1
        )
        Y = torch.bmm(self.dropout(attention_weights), values)
        return Y

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_size,
        num_hiddens,
        num_heads,
        dropout,
        key_size=None,
        value_size=None,
        bias=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        key_size = key_size or query_size
        value_size = value_size or query_size
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.w_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.w_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.w_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.w_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.w_k(keys), self.num_heads)
        values = self.transpose_qkv(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)

        output_concat = self.transpose_output(output, self.num_heads)
        Y = self.w_o(output_concat)
        return Y

    def transpose_qkv(self, X, num_heads):
        # Shape after transpose: (batch_size, num_heads, seq_len, num_hiddens/num_heads)
        X = X.view(X.shape[0], num_heads, -1).transpose(1, 2)
        return X.contiguous().view(X.shape[0], -1, X.shape[-1])

    def transpose_output(self, X, num_heads):
        # Shape after transpose: (batch_size, seq_len, num_heads, num_hiddens/num_heads)
        return X.contiguous().view(X.shape[0], -1)

class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x