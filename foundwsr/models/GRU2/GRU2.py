import torch
import torch.nn as nn
from .. import register_model
from ..base_model import BaseModel

@register_model("GRU2")
class GRU2(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.input_dim, args.signal_length,
                   args.num_classes, args.num_layers,
                   args.bidirectional)

    def __init__(self, in_dim, sig_len, num_classes, num_layers=2, bidirectional=False):
        super(GRU2, self).__init__()

        self.sig_len = sig_len
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=sig_len,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Linear(sig_len * self.num_directions, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.sig_len).to(x.device)
        out, _ = self.gru(x, h0)  # out: (B, T, sig_len * num_directions)
        last_output = out[:, -1, :]  # (B, sig_len * num_directions)
        logits = self.fc(last_output)  # (B, num_classes)
        return logits