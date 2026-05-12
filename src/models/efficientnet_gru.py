import torch
import torch.nn as nn
import timm

class DeepfakeDetector(nn.Module):
    """
    CNN-RNN hybrid model for deepfake detection.
    EfficientNet-B0 extracts spatial features, BiGRU processes temporal coherence.
    """
    def __init__(self, cnn_backbone='efficientnet_b0', hidden_dim=256, n_layers=2, n_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained EfficientNet-B0
        self.cnn = timm.create_model(cnn_backbone, pretrained=True)
        # We need to extract features before the classification head
        # For efficientnet_b0, it's 1280
        self.cnn_out_dim = self.cnn.num_features
        
        # Reset the CNN classification head to identity
        self.cnn.classifier = nn.Identity()
        
        # BiGRU for temporal sequence processing
        # input_size = 1280, hidden_size = 256, bidirectional = True
        self.gru = nn.GRU(
            input_size=self.cnn_out_dim, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        
        # Final classification head
        # bidirectional means hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        """
        Input x shape: (Batch, sequence_len, 3, 224, 224)
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # 1. Spatial Feature Extraction
        # Flatten (B, T) to B*T to process all frames through CNN at once
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x) # (B*T, 1280)
        
        # 2. Temporal Processing
        # Reshape back to (B, T, Features)
        features = features.view(batch_size, seq_len, -1)
        
        # GRU outputs: (output, h_n)
        # output is of shape (B, T, hidden_dim * 2)
        gru_out, _ = self.gru(features)
        
        # We take the mean across the temporal dimension (or the final state)
        # Taking the mean is robust to varying jitter locations
        temporal_features = torch.mean(gru_out, dim=1) # (B, hidden_dim * 2)
        
        # 3. Final Classification
        logits = self.classifier(temporal_features) # (B, n_classes)
        
        return logits
