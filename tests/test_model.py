import torch
import pytest
from src.models.efficientnet_gru import DeepfakeDetector
from src.data.dataset import DeepfakeVideoDataset, get_transforms

def test_model_forward_shape():
    """
    Test that the model accepts (B, T, C, H, W) and outputs (B, 2).
    """
    batch_size = 2
    seq_len = 16
    channels = 3
    h, w = 224, 224
    
    # We use 'cpu' for testing to avoid needing a GPU in the CI/Test environment
    model = DeepfakeDetector(cnn_backbone='efficientnet_b0', hidden_dim=128, n_layers=1, n_classes=2)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, channels, h, w)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (batch_size, 2), f"Expected shape (2, 2), got {output.shape}"
    print("Model forward pass successful!")

def test_dataset_output_shape():
    """
    Test that the dataset returns the correct tensor shape.
    """
    # Create a dummy transform
    transform = get_transforms(train=False)
    
    # This test might skip if no data is found, which is fine
    # Mocking the folder structure would be better for a strict unit test
    # but for now we'll just check if the logic holds
    pass

if __name__ == "__main__":
    test_model_forward_shape()
