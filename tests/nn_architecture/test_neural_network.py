import torch
import pytest

from src.nn_architecture.AlphaZeroNet import MLPEncoder
from src.nn_architecture.AlphaZeroNet import CNNEncoder



@pytest.mark.parametrize(
        "input_shape, output_shape, num_layers",
        [
            (4, 2, 10),
            (8, 2, 10),
            (8, 2, 0),
            (4,2000,10),
        ]
)
def test_mlp_encoder(input_shape, output_shape, num_layers):
    
    encoder = MLPEncoder(num_layers, input_shape, output_shape)

    x = torch.randn(input_shape) 

    y = encoder.forward(x)

    assert y.size(dim=0) == output_shape, f"Expected output {output_shape}, got {y.shape}"


def test_mlp_encoder_validation():
    with pytest.raises(ValueError) as excinfo:
        input_shape = 0
        output_shape = 10
        num_layers = 5

        encoder = MLPEncoder(num_layers, input_shape, output_shape)


def test_mlp_encoder_validation_output():
    with pytest.raises(ValueError) as excinfo:
        input_shape = 10
        output_shape = 0
        num_layers = 5

        encoder = MLPEncoder(num_layers, input_shape, output_shape)
    