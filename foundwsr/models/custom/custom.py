import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ..base_model import BaseModel
from .. import register_model

@register_model("custom")
class custom(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_dim, args.hidden_dim,
                   args.num_heads, args.num_layers,
                   args.dim_feedforward, args.signal_length)

    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_hidden_dim, signal_length):
        super().__init__()
        self.num_layers = num_layers
        time_dim = hidden_dim * 4
        self.time_embedding = TimeEmbedding(time_dim)
        self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.freq_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.time_encoder = nn.ModuleList()
        self.freq_encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.time_encoder.append(TimeLayer(hidden_dim, time_dim, num_heads, ff_hidden_dim, signal_length))
            self.freq_encoder.append(FreqLayer(hidden_dim, time_dim, ff_hidden_dim))

    def forward(self, x, fx, t):
        x = self.input_layer(x).transpose(1, 2)
        fx = self.freq_layer(fx).transpose(1, 2)
        t = self.time_embedding(t, None)
        for i in range(0, self.num_layers):
            x = self.time_encoder[i](x, fx, t)
            fx = self.freq_encoder[i](fx, x, t)
        return x, fx

class TimeLayer(nn.Module):
    def __init__(self, hidden_dim, time_dim, num_heads, ff_hidden_dim, signal_length):
        super(TimeLayer, self).__init__()
        self.attention = RelativePositionAttention(hidden_dim, num_heads, signal_length)
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

    def forward(self, x, y, t):
        t_ = self.time_emb(t).unsqueeze(1)
        x = x + self.gamma * y + t_
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
    def __init__(self, embed_dim, num_heads, signal_length):
        super(RelativePositionAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))

        pos = torch.arange(signal_length).float()
        sinusoid = torch.outer(pos, inv_freq)
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)

        self.sin = torch.stack([sin, sin], dim=-1).view(signal_length, self.head_dim)
        self.cos = torch.stack([cos, cos], dim=-1).view(signal_length, self.head_dim)

        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q * self.cos.unsqueeze(1).unsqueeze(0).to(Q.device) - torch.roll(Q, shifts=1, dims=-1) * self.sin.unsqueeze(1).unsqueeze(0).to(Q.device)
        K = K * self.cos.unsqueeze(1).unsqueeze(0).to(Q.device) - torch.roll(K, shifts=1, dims=-1) * self.sin.unsqueeze(1).unsqueeze(0).to(Q.device)
        q, k, v = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        # with torch.backends.cuda.sdp_kernel(enable_math=False):
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

    def forward(self, t, aug_label):
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

class FreqLayer(nn.Module):
    def __init__(self, hidden_dim, time_dim, ff_hidden_dim):
        super(FreqLayer, self).__init__()
        self.feed_forward1 = LocalRepresentation(hidden_dim)
        self.feed_forward2 = FCN(hidden_dim, ff_hidden_dim, hidden_dim)
        self.block = FreqBlock(hidden_dim, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )
        self.beta = nn.Parameter(-0.5 * torch.ones(1))
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.drop_path = DropPath(0.2)

    def forward(self, x, y, t):
        t_ = self.time_emb(t).unsqueeze(1)
        x = x + self.beta * y + t_
        x = x.transpose(1, 2)
        x = x + self.feed_forward1(x)
        x = self.norm1(x)
        x = x + self.block(x, t_)
        x = self.norm2(x)
        # x = x.transpose(1, 2)
        x = x + self.drop_path(self.feed_forward2(x))
        x = self.norm3(x).transpose(1, 2)

        return x

class FreqBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, reduction=16, dropout=0.2):
        super(FreqBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels * 2, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu = nn.SiLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # x = x.transpose(1, 2)
        out = self.conv1(x)
        out, gate = out.chunk(2, 1)
        out = out * gate.sigmoid()
        out = self.relu(self.bn1(out))
        out = out + t.transpose(1, 2)
        attention = self.fc(self.avg_pool(out).squeeze(-1)).unsqueeze(-1)
        out = out * self.sigmoid(attention)
        out = x + self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.relu(out)
        return out.contiguous()

class FCN(nn.Module):
    """
    Implementation of FCN layer with 1*1 convolutions.
    Input: tensor with shape [B, C, D]
    Output: tensor with shape [B, C, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LocalRepresentation(nn.Module):
    """
    Local Representation module for IQFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x