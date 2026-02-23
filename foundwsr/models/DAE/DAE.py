import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("DAE")
class DAE(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_dim, args.num_classes,
                   args.dropout)

    def __init__(self, in_dim=2, classes=11, dropout_rate=0.0):
        super(DAE, self).__init__()

        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)

        self.fc1 = nn.Linear(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, classes)

        self.time_distributed = nn.Linear(32, in_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, (hn1, cn1) = self.lstm1(x)
        x = self.dropout(x)
        x, (hn2, cn2) = self.lstm2(x)

        xc = self.fc1(hn2[-1])
        xc = self.bn1(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.bn2(xc)
        xc = F.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc3(xc)
        xc = F.softmax(xc, dim=-1)

        xd = self.time_distributed(x)

        return xc, xd