"""
Image Encoder module using ResNet architecture for medical image feature extraction.
"""
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder using ResNet18 for feature extraction.
    
    This encoder uses a pre-trained ResNet18 model with frozen weights and adds
    a fully connected layer to project features to the desired embedding size.
    """
    
    def __init__(self, embed_size):
        """
        Initialize the image encoder.
        
        Args:
            embed_size (int): Desired size of output feature embeddings
        """
        super(ImageEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Freeze ResNet parameters
        for param in resnet.parameters():
            param.requires_grad = False
            
        # Remove final FC layer
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Forward pass through the encoder.
        
        Args:
            images (torch.Tensor): Input images [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Image feature embeddings [batch_size, embed_size]
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features