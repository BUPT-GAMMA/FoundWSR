import torch
import torch.nn as nn
from .. import register_model
from ..base_model import BaseModel

@register_model("ResNet")
class ResNet(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_dim, args.num_classes,
                   args.signal_length, args.dropout)

    def __init__(self, in_dim, num_classes, sig_len, dropout):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(
            Residual_Stack(in_dim, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), pool=(2, 1)),
            Residual_Stack(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), pool=(2, 1)),
            Residual_Stack(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), pool=(2, 1)),
        )

        self.embeddings = nn.Sequential(
            nn.Linear(32 * sig_len, 2048),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),

            nn.Linear(2048, 512),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),

            nn.Linear(512, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1)).unsqueeze(-1)
        out = self.net(x)
        out = out.view(out.shape[0], -1)
        out = self.embeddings(out)
        return self.classifier(out)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, downsample=None):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True, momentum=0.99)
        self.downsample = downsample
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        return out


class Residual_Stack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), pool=(1, 1)):
        super(Residual_Stack, self).__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, bias=True)
        self.residual_unit1 = ResidualUnit(32, 32, kernel_size, stride, padding)
        self.residual_unit2 = ResidualUnit(32, 32, kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=pool, stride=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.residual_unit1(x)
        x = self.residual_unit2(x)

        return x