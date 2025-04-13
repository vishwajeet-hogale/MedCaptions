import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TransformerMedicalDecoder(nn.Module):
    def __init__(self, 
                 model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", 
                 image_embed_size=384,
                 freeze_base=True):
        """
        Transformer-based decoder for medical image captions
        
        Args:
            model_name (str): Pretrained transformer model to use
            image_embed_size (int): Size of image embedding from encoder
            freeze_base (bool): Whether to freeze base transformer model
        """
        super(TransformerMedicalDecoder, self).__init__()
        
        # Load pretrained model
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        
        # Freeze transformer base if requested
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
            
            # Unfreeze only the last 2 layers for fine-tuning
            trainable_layers = [self.transformer.encoder.layer[-1], 
                               self.transformer.encoder.layer[-2]]
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True
                    
            # Unfreeze pooler
            for param in self.transformer.pooler.parameters():
                param.requires_grad = True
        
        # Image feature projection to match transformer dimensions
        self.image_projection = nn.Sequential(
            nn.Linear(image_embed_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        # Fusion mechanism for combining image and text features
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Final projection to align with BERT embedding space
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size*2, 768)  # Match BERT embedding size
        )
        
    def forward(self, dummy_input, image_feats):
        """
        Forward pass to generate caption embeddings from image features
        
        Args:
            dummy_input: Placeholder to match LSTM interface (not used)
            image_feats: Image features from encoder [batch_size, embed_dim]
            
        Returns:
            Caption embedding in BERT space
        """
        # Project image features to transformer dimension
        image_proj = self.image_projection(image_feats)
        
        # Use the image embedding as a "prompt" for the transformer
        # We'll create a "dummy sequence" using the image embedding
        # This acts as context for the transformer to generate medical content
        batch_size = image_feats.shape[0]
        
        # Create input embeddings for transformer
        # Use the image embedding as first token, followed by zeros
        sequence_length = 4  # Short sequence is enough as we just want the pooled output
        dummy_sequence = torch.zeros(batch_size, sequence_length, 
                                   self.transformer.config.hidden_size).to(image_feats.device)
        
        # Set first position to the image embedding
        dummy_sequence[:, 0, :] = image_proj
        
        # Create attention mask (attending to all tokens)
        attention_mask = torch.ones(batch_size, sequence_length).to(image_feats.device)
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=dummy_sequence,
            attention_mask=attention_mask
        )
        
        # Get pooled output
        pooled_output = transformer_outputs.pooler_output
        
        # Fuse image features with transformer output
        fused_features = self.fusion_layer(
            torch.cat([image_proj, pooled_output], dim=1)
        )
        
        # Final projection to BERT embedding space
        caption_embedding = self.output_projection(fused_features)
        
        return caption_embedding
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="cpu"):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls()
        model.load_state_dict(checkpoint["decoder"])
        model.to(device)
        model.eval()
        return model 