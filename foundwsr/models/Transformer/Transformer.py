import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import register_model
from ..base_model import BaseModel

@register_model("Transformer")
class Transformer(BaseModel):
    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.in_dim, args.signal_length,
                   args.num_classes, args.num_layers,
                   args.transformer_dim, args.num_heads,
                   args.hidden_dim, args.dropout)

    def __init__(self, in_dim=2, sig_len=128, num_classes=11,
                 num_layers=2, transformer_dim=256,
                 num_heads=4, hidden_dim=512, dropout=0.1):
        super(Transformer, self).__init__()

        self.conv = nn.Conv2d(in_dim, transformer_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(transformer_dim)
        self.flatten_conv = nn.Flatten(start_dim=2)
        
        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.zeros(1, sig_len, transformer_dim))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully Connected Classification Layers
        self.fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.gru = nn.GRU(input_size=transformer_dim, hidden_size=transformer_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        # Input shape: B x 128 x 2
        x = x.permute(0,2,1).unsqueeze(-1) 
        x = self.conv(x) 
        x = self.bn(x)  
        x = F.tanh(x)
        x = self.flatten_conv(x)  # Flatten along the time dimension -> B x d_model x 128
        x = x.permute(0, 2, 1)  # Rearrange to B x 128 x d_model for Transformer

        # Add positional embedding
        x = x + self.positional_embedding # B x 128 x d_model

        # Transformer Encoding
        x = self.transformer_encoder(x)  # B x 128 x d_model
        x, _ = self.gru(x)
        x = x[:, -1, :]
        # Classification Head
        logits = self.fc(x)  # B x num_classes

        return logits