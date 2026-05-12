import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PretrainedDetector(nn.Module):
    """
    A Face Bounding Box Regressor built on a pre-trained ResNet18 backbone.
    This utilizes transfer learning to converge faster and achieve higher accuracy
    than training from scratch.
    """
    def __init__(self, freeze_backbone=False):
        super(PretrainedDetector, self).__init__()
        
        # Load pre-trained ResNet18
        # Using weights=ResNet18_Weights.DEFAULT gives us ImageNet weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Optionally freeze the backbone feature extractor so we only train the head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # ResNet18 outputs 512 features from its final fully connected layer before classification
        num_features = self.backbone.fc.in_features
        
        # Replace the ImageNet classification head with our Regression Head
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4), # Output: [x, y, w, h]
            nn.Sigmoid()       # Keeps coordinates between [0, 1] for relative bounding boxes
        )

    def forward(self, x):
        # The backbone includes our custom regression head at the end
        return self.backbone(x)

if __name__ == "__main__":
    model = PretrainedDetector()
    dummy = torch.randn(1, 3, 224, 224)
    print(f"Pretrained ResNet18 Model Output Shape: {model(dummy).shape}")
