import torch
import pytest

from src.nn_architecture.network_config import NetworkConfig
from src.nn_architecture.network_config import StemConfig
from src.nn_architecture.network_config import HeadConfig

from src.nn_architecture.AlphaZeroNet import MLPEncoder
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.nn_architecture.AlphaZeroNet import ResidualBlock

from src.nn_architecture.AlphaZeroNet import NetworkHead


@pytest.mark.parametrize(
    "num_layers, input_shape, output_shape",
    [
        (10, 4, 2),
        (10, 8, 2),
        (0, 8, 2),
        (10, 4, 2000),
    ],
)
def test_mlp_encoder(num_layers, input_shape, output_shape):
    encoder = MLPEncoder(num_layers, input_shape, output_shape)

    x = torch.randn(input_shape)

    y = encoder.forward(x)

    assert y.size(dim=0) == output_shape, (
        f"Expected output {output_shape}, got {y.shape}"
    )


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


"""
@pytest.mark.parametrize(
        "num_layers, input_shape, output_shape",
        [
            (10, (4, 4), 2),
            (10, 8, 2),
            (0, (8,8), 2),
            (10, (1000,4),2000),
        ]
)
def test_cnn_encoder(num_layers, input_shape, output_shape):
    
    encoder = CNNEncoder(num_layers, input_shape, output_shape)

    x = torch.randn(input_shape) 

    y = encoder.forward(x)

    assert y.size(dim=0) == output_shape, f"Expected output {output_shape}, got {y.shape}"


def test_cnn_encoder_validation():
    with pytest.raises(ValueError) as excinfo:
        input_shape = 0
        output_shape = 10
        num_layers = 5

        encoder = CNNEncoder(num_layers, input_shape, output_shape)


def test_cnn_encoder_validation_output():
    with pytest.raises(ValueError) as excinfo:
        input_shape = 10
        output_shape = 0
        num_layers = 5

        encoder = CNNEncoder(num_layers, input_shape, output_shape)   

"""


@pytest.mark.parametrize(
    "config",
    [
        # (NetworkConfig(encoder_type="cnn",input_shape=8,hidden_shape=20,output_shape=30,stem=StemConfig(num_residual_blocks=10), head=HeadConfig(hidden_blocks=5))),
        (
            NetworkConfig(
                encoder_type="mlp",
                input_shape=8,
                hidden_shape=20,
                legal_actions=5,
                num_layers=20,
                stem=StemConfig(num_residual_blocks=10, block_size=5),
                head=HeadConfig(hidden_blocks=5),
            )
        )
    ],
)
def test_alpha_zero_net(config):
    model = AlphaZeroNet(config)

    x = torch.randn(config.input_shape)

    policies, value = model.forward(x)

    assert policies.size(dim=0) == config.legal_actions, (
        f"Expected output {config.legal_actions}, got {policies.shape}"
    )
    assert value.size(dim=0) == 1, f"Expected output {1}, got {policies.shape}"


def test_alpha_zero_validate_encoder():
    with pytest.raises(ValueError) as excinfo:
        config = NetworkConfig(
            encoder_type="NOT_A_VALID_ENCODER",
            input_shape=10,
            hidden_shape=4,
            output_shape=10,
        )

        model = AlphaZeroNet(config)


def test_residual_block():
    block_size = 10
    hidden_dim = 128

    block = ResidualBlock(block_size, hidden_dim)

    x = torch.randn(block_size)

    y = block.forward(x)

    assert y.size(dim=0) == block_size, f"Expected output {block_size}, got {y.shape}"


def test_network_head():
    batch = 2
    legal_actions = 2
    input_shape = 128
    num_hidden_blocks = 5

    network_head = NetworkHead(legal_actions, input_shape, num_hidden_blocks)

    x = torch.randn(batch, input_shape)

    policy, value = network_head.forward(x)


    assert value.min() >= -1 and value.max() <= 1, (
        f"Expected value in -1 to 1, got {value}"
    )
    assert value.size() == (batch, 1), (
        f"Expected value size {(batch, 1)}, got {value.size()}"
    )
    assert policy.size() == (batch, legal_actions), (
        f"Expected policy size {(batch, legal_actions)}, got {policy.size()}"
    )
