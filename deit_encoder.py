import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class DeiTMedicalEncoder(nn.Module):
    def __init__(self, embed_size=768, model_name='deit_small_patch16_224', pretrained=True):
        """
        Medical image encoder based on DeiT (Data-efficient image Transformer)
        
        Args:
            embed_size: Output embedding size (768 to match BERT)
            model_name: DeiT model variant
            pretrained: Whether to use pretrained weights
        """
        super(DeiTMedicalEncoder, self).__init__()
        
        # Load pretrained DeiT model
        self.deit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        
        # Get feature dimension from DeiT
        if 'small' in model_name:
            self.feature_dim = 384  # DeiT-small hidden dimension
        elif 'tiny' in model_name:
            self.feature_dim = 192  # DeiT-tiny hidden dimension  
        elif 'base' in model_name:
            self.feature_dim = 768  # DeiT-base hidden dimension
        
        # Projection to desired embedding size
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embed_size),
            nn.LayerNorm(embed_size)
        )
        
        # Contrastive learning projection head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        # Unfreeze only the last transformer blocks for efficiency
        # Keep first layers frozen
        self._freeze_layers(unfreeze_last_n=4)
    
    def _freeze_layers(self, unfreeze_last_n=4):
        """Freeze all layers except the last n transformer blocks"""
        # Freeze patch embedding
        for param in self.deit.patch_embed.parameters():
            param.requires_grad = False
            
        # Freeze position embedding
        if hasattr(self.deit, 'pos_embed'):
            self.deit.pos_embed.requires_grad = False
        
        # Freeze class token
        if hasattr(self.deit, 'cls_token'):
            self.deit.cls_token.requires_grad = False
        
        # Freeze most transformer blocks, unfreeze only the last few
        total_blocks = len(self.deit.blocks)
        for i, block in enumerate(self.deit.blocks):
            # Only unfreeze the last n blocks
            if i >= total_blocks - unfreeze_last_n:
                for param in block.parameters():
                    param.requires_grad = True
            else:
                for param in block.parameters():
                    param.requires_grad = False
    
    def unfreeze_more_layers(self, additional_blocks=2):
        """Unfreeze more transformer blocks for progressive training"""
        total_blocks = len(self.deit.blocks)
        unfrozen_count = sum(1 for block in self.deit.blocks if next(block.parameters()).requires_grad)
        
        # Calculate which blocks to unfreeze next
        blocks_to_unfreeze = min(unfrozen_count + additional_blocks, total_blocks)
        
        # Unfreeze more blocks
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks - unfrozen_count):
            if i >= 0:  # Safety check
                for param in self.deit.blocks[i].parameters():
                    param.requires_grad = True
                print(f"Unfrozen block {i}")

    def forward(self, x, get_contrast_too=False):
        """
        Forward pass through DeiT encoder
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            get_contrast_too: Whether to return contrastive embedding
            
        Returns:
            Main embedding (and contrastive embedding if requested)
        """
        # Get DeiT features (CLS token output)
        features = self.deit(x)
        
        # Project to desired embedding dimension
        main_embed = self.projection(features)
        
        if get_contrast_too:
            # Get contrastive embedding
            contrast_embed = self.contrast_head(features)
            return main_embed, contrast_embed
        else:
            return main_embed 