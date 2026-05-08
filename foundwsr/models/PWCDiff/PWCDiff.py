import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ..base_model import BaseModel
from .. import register_model
from ...utils.channel import *
FUNCTION_MAP = {
    "rician": apply_rician,
    "cfo": apply_cfo,
    "sro": apply_sro,
    "awgn": apply_awgn,
    "sto": apply_sto,
    "rayleigh": apply_rayleigh,
    "multipath": apply_multipath,
    "nakagami": apply_nakagami
}

class ChannelGatingNet(nn.Module):
    def __init__(self, hidden_dim=64, n_channels=7):  # e.g., rician, cfo, sro
        super().__init__()
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=1),
            nn.GroupNorm(1, 16), nn.GELU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, groups=16),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.mlp = nn.Sequential(
            nn.Linear(32 * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, fx):
        time_feat = self.time_conv(x)
        freq_feat = self.freq_conv(fx)
        feat = torch.cat([time_feat, freq_feat], dim=-1)
        weights = self.mlp(feat)
        return weights

def stemIQ(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, D]
    """
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs//2 , kernel_size=5, stride=1, padding=2, groups=in_chs),
        nn.BatchNorm1d(out_chs//2),
        )

def stemFFT(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, 1, D]
    """
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs//2 , kernel_size=5, stride=1, padding=2, groups=in_chs),
        nn.BatchNorm1d(out_chs//2),
        nn.ReLU())

class Fusion(nn.Module):
    def __init__(self, input_chanel, drop):

        super().__init__()
        self.Conv = nn.Sequential( nn.Conv1d(input_chanel,input_chanel*2, 1),
                                  nn.BatchNorm1d(input_chanel*2),
                                  nn.GELU(),
                                  nn.Conv1d(input_chanel*2, input_chanel*2, 1),
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x, fx):
        fusion = self.Conv(torch.cat((x,fx), dim=1))
        return self.drop(fusion)

@register_model("PWCDiff")
class PWCDiff(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_dim, args.hidden_dim,
                   args.num_heads, args.num_layers,
                   args.dim_feedforward, args.channel_type)

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_hidden_dim, channel_type=None):
        super().__init__()
        self.num_layers = num_layers
        time_dim = hidden_dim * 4
        self.time_embedding = TimeEmbedding(time_dim)
        self.channel_type = channel_type
        self.channel = ChannelGatingNet(n_channels=len(self.channel_type))
        self.patch_embedIQ = stemIQ(input_dim, hidden_dim // 2)
        self.patch_embedFFT = stemFFT(input_dim, hidden_dim // 2)
        self.fusion = Fusion(hidden_dim // 2, 0.2)

        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder.append(TimeLayer(hidden_dim, time_dim, num_heads, ff_hidden_dim))

    @torch._dynamo.disable
    def get_input(self, x):
        x_complex = torch.view_as_complex(x.permute(0, 2, 1).contiguous())
        X_fft = torch.fft.fft(x_complex, dim=-1)
        fx = torch.stack([X_fft.real, X_fft.imag], dim=1)
        w = self.channel(x, fx)
        Y = X_fft
        for index, func in enumerate(self.channel_type):
            Y = FUNCTION_MAP[func](Y, w[:, index])

        return torch.stack([Y.real, Y.imag], dim=1)

    def forward(self, x, fx, t):
        x = self.patch_embedIQ(x)
        fx = self.patch_embedFFT(fx)
        x = self.fusion(x, fx).transpose(1, 2)
        t = self.time_embedding(t, None)
        for i in range(0, self.num_layers):
            x = self.encoder[i](x, t)
        return x

class TimeLayer(nn.Module):
    def __init__(self, hidden_dim, time_dim, num_heads, ff_hidden_dim):
        super(TimeLayer, self).__init__()
        self.attention = RelativePositionAttention(hidden_dim, num_heads)
        self.feed_forward1 = FeedForward(hidden_dim, ff_hidden_dim)
        self.feed_forward2 = FeedForward(hidden_dim, ff_hidden_dim)
        self.conv = DepthwiseConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.gamma = nn.Parameter(-0.5 * torch.ones(1))
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )
        self.drop_path = DropPath(0.2)

    def forward(self, x, t):
        t_ = self.time_emb(t).unsqueeze(1)
        x = x + t_
        x = x + self.feed_forward1(x)
        x = self.norm1(x)
        x = x + self.attention(x)
        x = self.norm2(x)
        x = x + self.conv(x, t_)
        x = self.norm3(x)
        x = x + self.drop_path(self.feed_forward2(x))
        x = self.norm4(x)

        return x

class RelativePositionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(RelativePositionAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.)
        self.norm = nn.LayerNorm(embed_dim)

        self.register_buffer(
            "inv_freq",
            1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        )

    def _get_rotary_embedding(self, seq_len, device):
        pos = torch.arange(seq_len, device=device).float()
        sinusoid = torch.outer(pos, self.inv_freq)
        return torch.sin(sinusoid), torch.cos(sinusoid)

    def apply_rotary_pos_emb(self, x, sin, cos):
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]

        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x2 * cos + x1 * sin
        return torch.cat([x1_rot, x2_rot], dim=-1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device

        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        sin, cos = self._get_rotary_embedding(seq_len, x.device)
        Q = self.apply_rotary_pos_emb(Q, sin, cos)
        K = self.apply_rotary_pos_emb(K, sin, cos)

        q, k, v = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.norm(output)
        output = self.output(output)
        output = self.dropout(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(DepthwiseConv, self).__init__()
        self.pointwise1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                                   groups=out_channels, padding=kernel_size // 2)
        self.pointwise2 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        x = x.transpose(1, 2)
        x = self.pointwise1(x)
        out, gate = x.chunk(2, 1)
        x = out * gate.sigmoid()
        x = x + t.transpose(1, 2)
        x = self.depthwise(x)
        x = self.bn(x)
        x = x * torch.sigmoid(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels, augment_dim=0):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.aug_emb = nn.Linear(augment_dim, self.n_channels // 4, bias=False) if augment_dim > 0 else None

        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t, aug_label=None):
        # Create sinusoidal position embeddings (same as those from the transformer)
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        if self.aug_emb is not None and aug_label is not None:
            emb += self.aug_emb(aug_label)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb
