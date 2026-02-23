import torch
import torch.nn as nn
from .. import register_model
from ..base_model import BaseModel

@register_model("MSNet")
class MSNet(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_channels,
                   args.num_classes)

    def __init__(self, in_channels, classes):
        super(MSNet, self).__init__()
        self.net = nn.Sequential(
            MultiScale(in_channels, stride=(2, 1)),
            MultiScale(128, 32, stride=(2, 1)),
            nn.AdaptiveAvgPool2d((4, 1))
        )
        self.embeddings = nn.Sequential(
            nn.Linear(32*4*4, 128),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1)).unsqueeze(-1)
        out = self.net(x)
        out = out.view(out.shape[0], -1)
        features = self.embeddings(out)
        logits = self.classifier(features)
        return logits


def ConvBNReLU(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, momentum=0.99),
        nn.ReLU(inplace=True)
    )


class MultiScale(nn.Module):
    def __init__(self, in_channels, channels=32, stride=(2, 1)):
        super(MultiScale, self).__init__()

        self.conv = ConvBNReLU(in_channels, channels, kernel_size=(3, 1), stride=stride, padding=(1, 0))

        self.branch1 = ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch2 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        )

        self.branch4 = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ConvBNReLU(channels, channels, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0))
        )

    def forward(self, x):
        x = self.conv(x)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out