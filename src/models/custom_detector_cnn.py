import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    The Core of Modern CNNs: The Skip Connection.
    Math: Output = Activation(Conv(x) + x)
    This allows the 'gradients' to flow through the network without dying.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # MATH: THE SKIP CONNECTION
        out = F.relu(out)
        return out

class CustomDetectorCNN(nn.Module):
    """
    A Deep Residual CNN for Bounding Box Regression.
    Designed for 90%+ Accuracy 'From Scratch'.
    """
    def __init__(self):
        super(CustomDetectorCNN, self).__init__()
        
        # Initial 'Entry' Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet-style Blocks
        # MATH: Each group focuses on a different scale of features.
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # Regression Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4), # [x, y, w, h]
            nn.Sigmoid() 
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # Entry
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Deep Residual Path
        out = self.layer1(out) # 224x224
        out = self.layer2(out) # 112x112
        out = self.layer3(out) # 56x56
        out = self.layer4(out) # 28x28
        
        # Flatten and Predict
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = CustomDetectorCNN()
    dummy = torch.randn(1, 3, 224, 224)
    print(f"Residual Model Output Shape: {model(dummy).shape}")
