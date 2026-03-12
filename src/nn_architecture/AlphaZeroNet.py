import torch
import torch.nn as nn
from src.nn_architecture.network_config import NetworkConfig


class AlphaZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        encoder_type = config.encoder_type
        match encoder_type:
            case "cnn":
                # TODO implement CNN
                self.model = CNNEncoder(
                    input_shape=config.input_shape,
                    output_shape=config.stem.block_size,
                    num_layers=config.num_layers,
                )
            case "mlp":
                self.model = MLPEncoder(
                    num_layers=config.num_layers,
                    input_shape=config.input_shape,
                    output_shape=config.stem.block_size,
                )
            case _:
                raise ValueError(f"Invalid encoder type: {encoder_type}")

        # stem
        self.common_blocks = nn.ModuleList(
            [
                ResidualBlock(config.stem.block_size)
                for i in range(config.stem.num_residual_blocks)
            ]
        )

        # output head
        self.head = NetworkHead(
            config.legal_actions, config.stem.block_size, config.head.hidden_blocks
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.model.forward(obs)
        for i, block in enumerate(self.common_blocks):
            x = block(x)

        # pred of values and policies
        policies, values = self.head.forward(x)
        return policies, values


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_shape: int,
        num_layers: int = 3,
        hidden_channels: int = 128,
    ):
        """
        Args:
            input_shape: (H, W, C) of the input observation
            output_shape: dimension of the output vector
            num_layers: number of convolutional layers in the stack
            hidden_channels: number of channels in each convolutional layer
        """
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError("CNNEncoder expects input_shape = (H, W, C)")

        height, width, channels = input_shape

        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("All input dimensions must be positive")
        if output_shape <= 0:
            raise ValueError("output_shape must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        conv_layers = []

        in_channels = channels
        for _ in range(num_layers):
            conv_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=hidden_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.LeakyReLU(),
                ]
            )
            in_channels = hidden_channels

        self.conv_stack = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(hidden_channels * height * width, output_shape)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: (B, H, W, C) or (H, W, C) tensor of observations
        Returns:
            (B, output_shape) tensor of encoded observations
        """

        if observation.dim() == 3:
            observation = observation.unsqueeze(0)

        if observation.dim() != 4:
            raise ValueError(
                f"Expected observation of shape (B,H,W,C) or (H,W,C), got {observation.shape}"
            )

        # Convert channels-last to channels-first for Conv2d
        x = observation.permute(0, 3, 1, 2)  # (B, C, H, W)

        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.output_layer(x)

        return x


class MLPEncoder(nn.Module):  # f : obs -> input
    def __init__(
        self,
        num_layers: int,
        input_shape: int,
        output_shape: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        if input_shape <= 0:
            raise ValueError
        if output_shape <= 0:
            raise ValueError

        self.input_layer = nn.Linear(input_shape, out_features=hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_shape)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        leakyrelu = nn.LeakyReLU()

        x = self.input_layer(observation)
        x = leakyrelu(x)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = leakyrelu(x)
        x = self.output_layer(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, block_size: int, hidden_dim: int = 128):
        super().__init__()

        self.layer1 = nn.Linear(block_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leakyrelu = nn.LeakyReLU()

        activation = self.layer1(x)
        activation = leakyrelu(activation)

        activation = self.layer2(activation)
        activation = leakyrelu(activation)

        activation += x
        activation = leakyrelu(activation)

        return activation


class NetworkHead(nn.Module):
    def __init__(self, legal_actions, input_shape, num_hidden_blocks):
        super().__init__()
        self.common_block = nn.ModuleList(
            [ResidualBlock(input_shape) for i in range(num_hidden_blocks)]
        )

        self.value_head = nn.Linear(input_shape, 1)
        self.policy_head = nn.Linear(input_shape, legal_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tanh = nn.Tanh()
        value = self.value_head(x)
        value = tanh(value)

        softmax = nn.Softmax()
        policy_logits = self.policy_head(x)
        policy_logits = softmax(policy_logits)

        # [B,A], [B,1]
        return policy_logits, value
