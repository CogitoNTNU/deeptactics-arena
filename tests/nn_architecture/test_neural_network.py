from src.environment import build_environment
import torch
import pytest

from src.nn_architecture.network_config import NetworkConfig
from src.nn_architecture.network_config import StemConfig
from src.nn_architecture.network_config import HeadConfig
from src.nn_architecture.AlphaZeroNet import CNNEncoder, MLPEncoder
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
    with pytest.raises(ValueError):
        input_shape = 0
        output_shape = 10
        num_layers = 5

        MLPEncoder(num_layers, input_shape, output_shape)


def test_mlp_encoder_validation_output():
    with pytest.raises(ValueError):
        input_shape = 10
        output_shape = 0
        num_layers = 5

        MLPEncoder(num_layers, input_shape, output_shape)


@pytest.mark.parametrize("env_name", ["chess", "connect_four", "tic_tac_toe"])
def test_cnn_encoder_for_all_games(env_name):
    env = build_environment(env_name)

    output_shape: int = 128
    num_layers: int = 3
    hidden_channels: int = 2
    env.reset()

    agent = env.agents[0]
    obs_shape = env.observation_space(agent)["observation"]._shape
    encoder = CNNEncoder(obs_shape, output_shape, num_layers, hidden_channels)

    x = torch.randn(*obs_shape)
    y = encoder.forward(x)

    assert y.shape[-1] == output_shape, (
        f"Expected output last dim {output_shape}, got {y.shape}"
    )


@pytest.mark.parametrize(
    "config",
    [
        (
            NetworkConfig(
                encoder_type="cnn",
                input_shape=(8, 8, 111),
                hidden_shape=20,
                legal_actions=5,
                num_layers=20,
                stem=StemConfig(num_residual_blocks=10, block_size=5),
                head=HeadConfig(hidden_blocks=5),
            )
        ),
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
        ),
    ],
)
def test_alpha_zero_net(config):
    model = AlphaZeroNet(config)

    x = torch.randn(config.input_shape)

    policies, value = model.forward(x)

    assert policies.shape[-1] == config.legal_actions, (
        f"Expected output {config.legal_actions}, got {policies.shape}"
    )
    assert value.shape[-1] == 1, f"Expected output {1}, got {value.shape}"


def test_batched_alpha_zero_net():
    config = NetworkConfig(
        encoder_type="cnn",
        input_shape=(8, 8, 111),
        hidden_shape=20,
        legal_actions=5,
        num_layers=20,
        stem=StemConfig(num_residual_blocks=10, block_size=5),
        head=HeadConfig(hidden_blocks=5),
    )
    model = AlphaZeroNet(config)

    batched_obs = torch.randn(2, *config.input_shape)

    policies, value = model.forward(batched_obs)

    assert policies.shape[-1] == config.legal_actions, (
        f"Expected output {config.legal_actions}, got {policies.shape}"
    )
    assert value.shape[-1] == 1, f"Expected output {1}, got {value.shape}"


def test_alpha_zero_validate_encoder():
    with pytest.raises(ValueError):
        config = NetworkConfig(
            encoder_type="NOT_A_VALID_ENCODER",
            input_shape=10,
            hidden_shape=4,
            output_shape=10,
        )

        AlphaZeroNet(config)


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
