import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("CGDNN")
class CGDNN(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_channels, args.output_channels, args.num_classes,
                   args.dropout, args.signal_length, args.hidden_dim, args.kernel_size)

    def __init__(self, input_channels=1, output_channels=50,
                 classes=11, dropout_rate=0.5, sig_len=128,
                 hidden_dim=256, kernel_size=6):
        super(CGDNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(kernel_size, 1))
        self.gaussian_dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(kernel_size, 1))
        self.conv3 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=(kernel_size, 1))
        self.gru = nn.GRU(input_size=4 * sig_len - 8 * kernel_size + 8, hidden_size=output_channels, batch_first=True)

        self.fc1 = nn.Linear(output_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = F.relu(self.conv1(x))
        x1 = self.gaussian_dropout(x1)

        x2 = F.relu(self.conv2(x1))
        x2 = self.gaussian_dropout(x2)

        x3 = F.relu(self.conv3(x2))
        x3 = self.gaussian_dropout(x3)

        x_concat = torch.cat([x1, x3], dim=-2)
        x_reshaped = x_concat.view(x_concat.size(0), 50, -1)
        x_gru, _ = self.gru(x_reshaped) 
        x_fc = F.relu(self.fc1(x_gru[:, -1, :])) 
        x_fc = self.gaussian_dropout(x_fc)
        x_out = self.fc2(x_fc)

        return x_out