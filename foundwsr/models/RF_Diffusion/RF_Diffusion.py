import math
import torch
import torch.nn as nn
from ...utils import SignalDiffusion
from ...utils import init_weight_xavier, init_weight_zero, init_weight_norm
from ...complex import complex_module as cm
from ..base_model import BaseModel
from .. import register_model

@torch.compile
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

@register_model("RF_Diffusion")
class RF_Diffusion(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_dim, args.hidden_dim,
                   args.embed_dim, args.num_heads,
                   args.dropout, args.mlp_ratio,
                   args.max_step, args.num_block, args.batch_size,
                   args.extra_dim, args.blur_noise,
                   args.min_noise, args.max_noise)

    def __init__(self, input_dim, hidden_dim, embed_dim, num_heads, dropout, mlp_ratio,
                 max_step, num_block, sample_rate, extra_dim, blur_noise, min_noise, max_noise):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mlp_ratio = mlp_ratio
        self.diffusion = SignalDiffusion(sample_rate, extra_dim, max_step,
                                         blur_noise, min_noise, max_noise)
        self.p_embed = PositionEmbedding(
            sample_rate, input_dim, hidden_dim)
        self.t_embed = DiffusionEmbedding(
            max_step, embed_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.dropout, self.mlp_ratio) for _ in range(num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)
    
    def forward(self, x, t):
        # t = torch.tensor([99]).to("cpu")
        # from ...utils import plot_signal

        # t = t.to("cpu")
        # x = self.diffusion.degrade_fn(x, t)

        # plot_signal(x[0].cpu().detach().numpy(), "./")
        # import sys
        # sys.exit()
        x = self.p_embed(x)
        t = self.t_embed(t)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)

        return x

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_len, hidden_dim), persistent=False)
        self.projection = cm.ComplexLinear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)

    def forward(self, x): 
        x = self.projection(x)
        return cm.complex_mul(x, self.embedding.to(x.device))

    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len)
                                  * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.attn = cm.ComplexMultiHeadAttention(
            hidden_dim, hidden_dim, num_heads, dropout, bias=True, **block_kwargs)
        self.norm2 = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            cm.ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
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


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = cm.NaiveComplexLayerNorm(
            hidden_dim, eps=1e-6, elementwise_affine=False)
        self.linear = cm.ComplexLinear(hidden_dim, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x).squeeze(0)
        return x

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            cm.ComplexLinear(embed_dim, hidden_dim, bias=True),
            cm.ComplexSiLU(),
            cm.ComplexLinear(hidden_dim, hidden_dim, bias=True),
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
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step)
                                  * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table

