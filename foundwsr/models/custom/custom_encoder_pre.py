import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..base_model import BaseModel
from .. import register_model

@register_model("custom_encoder_pre")
class custom_encoder(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_dim, args.hidden_dim,
                   args.num_heads, args.num_layers,
                   args.dim_feedforward)
    
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, ff_hidden_dim, max_len=1024):
        super().__init__()
        self.num_layers = num_layers
        time_dim = hidden_dim * 4
        self.time_embedding = TimeEmbedding(time_dim)
        self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder.append(SpectrumFMLayer(hidden_dim, time_dim, num_heads, ff_hidden_dim, max_len))

    def forward(self, x, t, mask=None):
        x = self.input_layer(x).permute(0, 2, 1)
        if t is not None:
            t = self.time_embedding(t, None)
        for i in range(0, self.num_layers):
            x = self.encoder[i](x, t, mask)

        return x


class SpectrumFMLayer(BaseModel):
    def __init__(self, hidden_dim, time_dim, num_heads, ff_hidden_dim, max_len):
        super(SpectrumFMLayer, self).__init__()
        self.attention = RelativePositionAttention(hidden_dim, num_heads, max_len)
        self.feed_forward1 = FeedForward(hidden_dim, ff_hidden_dim)
        self.feed_forward2 = FeedForward(hidden_dim, ff_hidden_dim)
        self.conv = DepthwiseConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim)
        )
     
    def forward(self, x, t, mask):
        if t is not None:
            t_ = self.time_emb(t).unsqueeze(1)
            x = x + t_
        x = x + 0.5 * self.feed_forward1(x)
        x = self.norm1(x)
        x = x + self.attention(x, mask)
        x = self.norm2(x)
        x = x + self.conv(x)
        x = self.norm3(x)
        x = x + 0.5 * self.feed_forward2(x)
        x = self.norm4(x)
        return x


class RelativePositionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_len=512):
        super(RelativePositionAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_len = max_len

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.relative_positions = nn.Parameter(torch.randn(2 * max_len - 1, num_heads))  # shape: [2*max_len-1, num_heads]

        self.output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x)  # [batch_size, seq_len, embed_dim]
        K = self.key(x)    # [batch_size, seq_len, embed_dim]
        V = self.value(x)  # [batch_size, seq_len, embed_dim]

        Q = Q.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        K = K.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)
        V = V.view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads)

        attention_scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = attention_scores / (self.embed_dim // self.num_heads) ** 0.5

        position_indices = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)  # [seq_len, seq_len]
        position_indices = position_indices + self.max_len - 1
        position_indices = position_indices.clamp(min=0, max=2 * self.max_len - 2)
        relative_position_embedding = self.relative_positions[position_indices]  # [seq_len, seq_len, num_heads]
        relative_position_embedding = relative_position_embedding.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        relative_position_embedding = relative_position_embedding.unsqueeze(0)
        attention_scores += relative_position_embedding 
        if mask is not None:
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            mask_value = -torch.finfo(attention_scores.dtype).max
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)

        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
       
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)  # [batch_size, seq_len, num_heads, embed_dim // num_heads]

        output = output.contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.output(output)
        output = self.dropout(output)

        return output

class FeedForward(nn.Module):
    def __init__(self, intput_dim, hidden_dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(intput_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, intput_dim)
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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pointwise1(x)
        out, gate = x.chunk(2, 1)
        x = out * gate.sigmoid()
        x = self.depthwise(x)
        x = self.bn(x)
        x = x * torch.sigmoid(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
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