import torch
import torch.nn as nn


class CaptionLSTM(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Image feature transformation for hidden state initialization
        self.image_to_hidden = nn.Sequential(
            nn.Linear(384, hidden_size),
            nn.Tanh()
        )

        # Direct BERT embedding processing (no projection)
        self.lstm = nn.LSTM(
            input_size=768,  # Direct BERT dimension
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Enhanced output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size*2, 768)  # Matching BERT output
        )

    def forward(self, x, image_feats):
        # x: [batch, seq_len, 768] (BERT embeddings)
        # image_feats: [batch, 256] (CNN features)

        # Initialize hidden state from image features
        h0 = self.image_to_hidden(image_feats)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_size]
        c0 = torch.zeros_like(h0)

        # LSTM processing
        out, _ = self.lstm(x, (h0, c0))

        # Process final output
        return self.fc(out[:, -1, :])  # Last timestep only