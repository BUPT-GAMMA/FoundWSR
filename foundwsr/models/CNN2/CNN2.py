import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("CNN2")
class CNN2(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_channels, args.num_classes,
                   args.dropout, args.signal_length)

    def __init__(self, in_channels, num_classes, dropout, sig_len=128):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=(5, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(5, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 1), padding='same')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same')

        self.pool = nn.MaxPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128 * int(sig_len // 16), 128)
        self.fc2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x