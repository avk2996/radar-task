import torch
import pytest
# Assume your model class is named RadarSegmentationModel
# from notebook_module import RadarSegmentationModel 

def test_model_output_shape():
    model = RadarSegmentationModel(num_classes=5)
    dummy_input = torch.randn(1, 6, 50, 181) # Batch size 1, 6 channels
    output = model(dummy_input)
    assert output.shape == (1, 5, 50, 181), "Output dimensions are incorrect"

def test_loss_function():
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = torch.randn(1, 5, 50, 181)
    target = torch.randint(0, 5, (1, 50, 181))
    loss = loss_fn(logits, target)
    assert loss > 0