"""
LSTM-based decoder for medical image caption generation.
"""
import torch
import torch.nn as nn


class CaptionLSTM(nn.Module):
    """
    LSTM decoder for generating caption embeddings from image features.
    
    This module processes image features through an LSTM network to generate
    caption embeddings in BERT embedding space.
    """
    
    def __init__(self, hidden_size=1024, num_layers=2):
        """
        Initialize the LSTM caption decoder.
        
        Args:
            hidden_size (int): Hidden size of LSTM layers
            num_layers (int): Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Image feature transformation for hidden state initialization
        self.image_to_hidden = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.Tanh()
        )

        # LSTM for processing BERT embeddings
        self.lstm = nn.LSTM(
            input_size=768,  # BERT embedding dimension
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Output projection layer to BERT embedding space
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, 768)  # Match BERT output dimension
        )

    def forward(self, x, image_feats):
        """
        Forward pass through the LSTM decoder.
        
        Args:
            x (torch.Tensor): Input embeddings [batch, seq_len, 768]
            image_feats (torch.Tensor): Image features [batch, embed_dim]
            
        Returns:
            torch.Tensor: Caption embeddings [batch, 768]
        """
        # Initialize hidden state from image features
        h0 = self.image_to_hidden(image_feats)
        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_size]
        c0 = torch.zeros_like(h0)

        # LSTM processing
        out, _ = self.lstm(x, (h0, c0))

        # Process final output (last timestep)
        return self.fc(out[:, -1, :])