import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("MCNet")
class MCNet(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_dim, args.num_classes, 
                   args.signal_length, args.dropout)

    def __init__(self, in_dim, num_classes, sig_len, dropout):
        super(MCNet, self).__init__()

        self.conv1_1 = nn.Conv2d(in_dim, 64, kernel_size=(3, 7), stride=(1, 2), padding=(1, 3))
        self.pool1_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Preblock
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Skip connection
        self.conv111 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))
        self.pool2_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        # Mblockp1
        self.pool3_1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_1 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool3_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblock2
        self.conv4_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv4_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv4_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv4_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Mblockp3
        self.conv5_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv5_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool5_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv5_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblockp4
        self.conv6_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv6_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv6_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv6_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Mblockp5
        self.conv7_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv7_2 = nn.Conv2d(32, 48, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.pool7_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv7_3 = nn.Conv2d(32, 48, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.conv7_4 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 2), padding=(0, 0))

        # Mblockp6
        self.conv8_1 = nn.Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv8_2 = nn.Conv2d(32, 96, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.conv8_3 = nn.Conv2d(32, 96, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv8_4 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Output layers
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(384 * int(sig_len // 2), num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv2D
        x = F.relu(self.conv1_1(x))
        x = self.pool1_1(x)

        x2 = F.relu(self.conv2_1(x))
        x2 = self.pool2_1(x2)
        x22 = F.relu(self.conv2_2(x))
        x222 = torch.cat([x2, x22], dim=1)

        xx1 = F.relu(self.conv111(x222))
        xx1 = self.pool2_2(xx1)

        x3 = self.pool3_1(x222)
        x3 = F.relu(self.conv3_1(x3))
        x31 = F.relu(self.conv3_2(x3))
        x31 = self.pool3_2(x31)
        x32 = F.relu(self.conv3_3(x3))
        x33 = F.relu(self.conv3_4(x3))
        x31 = torch.cat([x31, x32], dim=1)
        x333 = torch.cat([x33, x31], dim=1)

        add1 = x333 + xx1

        x4 = F.relu(self.conv4_1(add1))
        x41 = F.relu(self.conv4_2(x4))
        x42 = F.relu(self.conv4_3(x4))
        x43 = F.relu(self.conv4_4(x4))
        x41 = torch.cat([x41, x42], dim=1)
        x444 = torch.cat([x43, x41], dim=1)

        add2 = x444 + add1

        x5 = F.relu(self.conv5_1(add2))
        x51 = F.relu(self.conv5_2(x5))
        x51 = self.pool5_2(x51)
        x52 = F.relu(self.conv5_3(x5))
        x53 = F.relu(self.conv5_4(x5))
        x51 = torch.cat([x51, x52], dim=1)
        x555 = torch.cat([x53, x51], dim=1)

        ad3 = self.pool5_2(add2)
        add3 = x555 + ad3

        x6 = F.relu(self.conv6_1(add3))
        x61 = F.relu(self.conv6_2(x6))
        x62 = F.relu(self.conv6_3(x6))
        x63 = F.relu(self.conv6_4(x6))
        x61 = torch.cat([x61, x62], dim=1)
        x666 = torch.cat([x63, x61], dim=1)

        add4 = x666 + add3

        x7 = F.relu(self.conv7_1(add4))
        x71 = F.relu(self.conv7_2(x7))
        x71 = self.pool7_2(x71)
        x72 = F.relu(self.conv7_3(x7))
        x73 = F.relu(self.conv7_4(x7))
        x71 = torch.cat([x71, x72], dim=1)
        x777 = torch.cat([x73, x71], dim=1)

        ad5 = self.pool7_2(add4)
        add5 = x777 + ad5

        x8 = F.relu(self.conv8_1(add5))
        x81 = F.relu(self.conv8_2(x8))
        x82 = F.relu(self.conv8_3(x8))
        x83 = F.relu(self.conv8_4(x8))
        x81 = torch.cat([x81, x82], dim=1)
        x888 = torch.cat([x83, x81], dim=1)

        x_con = torch.cat([x888, add5], dim=1)
        xout = self.avg_pool(x_con)
        xout = self.dropout(xout)
        xout = torch.flatten(xout, 1)
        xout = self.fc(xout)

        return xout