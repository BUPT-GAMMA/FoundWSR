import math
import torch
from torch import nn
from .. import register_model
from ..base_model import BaseModel

@register_model("DDAE_Network")
class Network(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.signal_length, args.network["n_channels"],
                   args.network["ch_mults"], args.network["is_attn"], None,
                   args.network["dropout"], args.network["n_blocks"],
                   args.network["use_res_for_updown"])

    def __init__(self, signal_length, n_channels = 256,
                 ch_mults = (1, 2, 2, 2),
                 is_attn = (False, True, False, False),
                 attn_channels_per_head = None,
                 dropout = 0.1,
                 n_blocks = 2,
                 use_res_for_updown = False,
                 augment_dim = 0):
        super().__init__()

        n_resolutions = len(ch_mults)

        self.input_layer = nn.Conv1d(2, n_channels, kernel_size=1)

        # Embedding layers (time & augment)
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels, augment_dim)

        # Down stages
        down = []
        in_channels = n_channels
        h_channels = []
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks` at the same resolution
            down.append(ResAttBlock(in_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
            h_channels.append(out_channels)
            for _ in range(n_blocks - 1):
                down.append(ResAttBlock(out_channels, out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
                h_channels.append(out_channels)
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                if use_res_for_updown:
                    down.append(DownsampleRes(out_channels, time_channels, dropout))
                else:
                    down.append(Downsample(out_channels))
                # h_channels.append(out_channels)
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, time_channels, attn_channels_per_head, dropout)

        # Up stages
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            # Number of output channels at this resolution
            out_channels = n_channels * ch_mults[i]
            # `n_blocks + 1` at the same resolution
            for _ in range(n_blocks):
                up.append(ResAttBlock(in_channels + h_channels.pop(), out_channels, time_channels, is_attn[i], attn_channels_per_head, dropout))
                in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                if use_res_for_updown:
                    up.append(UpsampleRes(out_channels, time_channels, dropout))
                else:
                    up.append(Upsample(out_channels))
        assert not h_channels
        self.up = nn.ModuleList(up)

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv1d(out_channels, 2, kernel_size=3, padding=1) # I do not think this is better

    def forward(self, x, t, aug=None, ret_activation=False):
        if not ret_activation:
            return self.forward_core(x, t, aug)

        activation = {}
        def namedHook(name):
            def hook(module, input, output):
                activation[name] = output
            return hook
        hooks = {}
        no = 0
        for blk in self.up:
            if isinstance(blk, ResAttBlock):
                no += 1
                name = f'out_{no}'
                hooks[name] = blk.register_forward_hook(namedHook(name))
        result = self.forward_core(x, t, aug)
        for name in hooks:
            hooks[name].remove()
        return result, activation

    def forward_core(self, x, t, aug):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        t = self.time_emb(t, aug)
        x = self.input_layer(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]

        for m in self.down:
            if isinstance(m, Downsample):
                x = m(x)
            elif isinstance(m, DownsampleRes):
                x = m(x, t)
            else:
                x = m(x, t).contiguous()
                h.append(x)

        x = self.middle(x, t).contiguous()

        for i, m in enumerate(self.up):
            if isinstance(m, Upsample):
                x = m(x)
            elif isinstance(m, UpsampleRes):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t).contiguous()

        return self.final(self.act(self.norm(x.permute(0, 2, 1))).permute(0, 2, 1))

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels, augment_dim):
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


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, d_k):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        n_heads = n_channels // d_k

        self.norm = nn.LayerNorm(n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)

        self.scale = 1 / math.sqrt(math.sqrt(d_k))
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x):
        """
        * `x` has shape `[batch_size, in_channels, dim]`
        """
        batch_size, n_channels, dim = x.shape
        # Normalize and rearrange to `[batch_size, seq, n_channels]`
        h = self.norm(x.permute(0, 2, 1)).view(batch_size, -1, n_channels)

        # {q, k, v} all have a shape of `[batch_size, seq, n_heads, d_k]`
        qkv = self.projection(h).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q * self.scale, k * self.scale) # More stable with f16 than dividing afterwards
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape to `[batch_size, seq, n_heads * d_k]` and transform to `[batch_size, seq, n_channels]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res.permute(0, 2, 1).view(batch_size, n_channels, dim)
        return res + x


class Upsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            return self.conv(x)
        else:
            return x


class Downsample(nn.Module):
    def __init__(self, n_channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.LayerNorm(out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for embeddings
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        # BigGAN style: use resblock for up/downsampling
        self.updown = up or down
        if up:
            self.h_layer = Upsample(in_channels, use_conv=False)
            self.x_layer = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_layer = Downsample(in_channels, use_conv=False)
            self.x_layer = Downsample(in_channels, use_conv=False)
        else:
            self.h_layer = self.x_layer = nn.Identity()

    def forward(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, signal_length]`
        * `t` has shape `[batch_size, time_channels]`
        """
        if self.updown:
            h = self.conv1(self.h_layer(self.act1(self.norm1(x.permute(0, 2, 1))).permute(0, 2, 1)))
            x = self.x_layer(x)
        else:
            h = self.conv1(self.act1(self.norm1(x.permute(0, 2, 1))).permute(0, 2, 1))

        # Adaptive Group Normalization
        t_ = self.time_emb(t)[:, :, None]
        h = h + t_

        h = self.conv2(self.act2(self.norm2(h.permute(0, 2, 1))).permute(0, 2, 1))
        return h + self.shortcut(x)


class ResAttBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn, attn_channels_per_head, dropout):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, dropout=dropout)
        if has_attn:
            self.attn = AttentionBlock(out_channels, attn_channels_per_head)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels, attn_channels_per_head, dropout):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)
        self.attn = AttentionBlock(n_channels, attn_channels_per_head)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UpsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, up=True)

    def forward(self, x, t):
        return self.op(x, t)


class DownsampleRes(nn.Module):
    def __init__(self, n_channels, time_channels, dropout):
        super().__init__()
        self.op = ResidualBlock(n_channels, n_channels, time_channels, dropout=dropout, down=True)

    def forward(self, x, t):
        return self.op(x, t)
