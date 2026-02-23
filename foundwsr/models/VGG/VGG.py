import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("VGG")
class VGG(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_dim, args.num_classes,
                   args.num_layers, args.signal_length,
                   args.dropout)

    def __init__(self, in_dim, num_classes, num_layers, sig_len, dropout):
        super(VGG, self).__init__()
        self.base = nn.ModuleList()
        self.base.extend(conv_bn_relu_maxpool(in_dim, 64, kernel_size=(3, 2), stride=(1, 2), padding=(1, 1), pool=(2, 1)))
        dim = sig_len // 2
        for _ in (1, num_layers):
            self.base.extend(conv_bn_relu_maxpool(64, 64, kernel_size=(1, 2), stride=(1, 2), padding=(1, 1)))
            dim = dim + 2
            dim = dim // 2

        self.embeddings = nn.Sequential(
            nn.Linear(dim * 64, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.AlphaDropout(dropout)
        )
        self.classifier = nn.Linear(128, num_classes)

        # self.apply(_weights_init)

    def forward(self, x):
        x = torch.permute(x, (0,2,1)).unsqueeze(-1)
        for l in self.base:
            x = l(x)
        x = x.view(x.shape[0], -1)
        x = self.embeddings(x)
        return self.classifier(x)

def conv_bn_relu_maxpool(in_channels, out_channels, kernel_size,  stride, padding, pool=(2, 1)):
    return nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool, stride=(2, 1))])
