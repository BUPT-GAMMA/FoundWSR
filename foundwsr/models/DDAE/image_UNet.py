import torch
from torch import nn
from .block import GroupNorm32, TimeEmbedding, AttentionBlock, Upsample, Downsample
from .. import register_model
from ..base_model import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1, up=False, down=False):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `dropout` is the dropout rate
        """
        super().__init__()
        self.norm1 = GroupNorm32(in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = GroupNorm32(out_channels)
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
            self.h_upd = Upsample(in_channels, use_conv=False)
            self.x_upd = Upsample(in_channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(in_channels, use_conv=False)
            self.x_upd = Downsample(in_channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

    def forward(self, x, t):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        if self.updown:
            h = self.conv1(self.h_upd(self.act1(self.norm1(x))))
            x = self.x_upd(x)
        else:
            h = self.conv1(self.act1(self.norm1(x)))

        # Adaptive Group Normalization
        t_ = self.time_emb(t)[:, :, None]
        h = h + t_

        h = self.conv2(self.act2(self.norm2(h)))
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
 
@register_model("DDAE_UNet")
class UNet(BaseModel):
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

        self.image_proj = nn.Conv2d(1, n_channels, kernel_size=1, padding=1)

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

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(out_channels, 1, kernel_size=1, padding=1)

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
        x = self.image_proj(x)

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

        return self.final(self.act(self.norm(x)))

