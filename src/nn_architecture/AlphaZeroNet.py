import torch
import torch.nn as nn
from src.nn_architecture.network_config import NetworkConfig


class AlphaZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()

        encoder_type = config.encoder_type
        match encoder_type:
            case "cnn":
                assert isinstance(config.input_shape, list)
                height, width, _ = config.input_shape
                input_shape = tuple(config.input_shape)
                self.model = CNNEncoder(
                    input_shape=input_shape,
                    output_channels=config.stem.block_size,
                    num_layers=config.num_layers,
                    kernel_size=config.kernel_size,
                )
                # Determine actual spatial dims after encoder (kernel padding may change H/W)
                with torch.no_grad():
                    _dummy = torch.zeros(1, height, width, input_shape[2])
                    _, _, enc_h, enc_w = self.model(_dummy).shape
                self.common_blocks = nn.ModuleList(
                    [
                        ConvResidualBlock(config.stem.block_size)
                        for _ in range(config.stem.num_residual_blocks)
                    ]
                )
                self.head = ConvNetworkHead(
                    legal_actions=config.legal_actions,
                    channels=config.stem.block_size,
                    height=enc_h,
                    width=enc_w,
                )
            case "mlp":
                self.model = MLPEncoder(
                    num_layers=config.num_layers,
                    input_shape=config.input_shape,
                    output_shape=config.stem.block_size,
                )
                self.common_blocks = nn.ModuleList(
                    [
                        ResidualBlock(config.stem.block_size)
                        for _ in range(config.stem.num_residual_blocks)
                    ]
                )
                self.head = LinearNetworkHead(
                    config.legal_actions,
                    config.stem.block_size,
                    config.head.hidden_blocks,
                )
            case _:
                raise ValueError(f"Invalid encoder type: {encoder_type}")

    def forward(
        self, obs: torch.Tensor, action_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Conv path requires batch dim throughout; handle unbatched (H,W,C) input here
        was_unbatched = obs.dim() == 3
        if was_unbatched:
            obs = obs.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        x = self.model(obs)
        for block in self.common_blocks:
            x = block(x)
        policies, values = self.head(x, action_mask=action_mask)

        if was_unbatched:
            policies = policies.squeeze(0)
            values = values.squeeze(0)

        return policies, values


class CNNEncoder(nn.Module):
    """Initial conv block: (B, H, W, C) → (B, output_channels, H, W)."""

    def __init__(
        self,
        input_shape: tuple[int, ...] | list[int],
        output_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()

        if len(input_shape) != 3:
            raise ValueError("CNNEncoder expects input_shape = (H, W, C)")

        height, width, in_channels = input_shape

        if height <= 0 or width <= 0 or in_channels <= 0:
            raise ValueError("All input dimensions must be positive")
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        layers = []
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels if i == 0 else output_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(),
                ]
            )

        self.conv_stack = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: (B, H, W, C) or (H, W, C)
        Returns:
            (B, output_channels, H, W)
        """
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)

        if observation.dim() != 4:
            raise ValueError(f"Expected (H,W,C) or (B,H,W,C), got {observation.shape}")

        x = observation.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        return self.conv_stack(x)


class ConvResidualBlock(nn.Module):
    """AlphaZero-style convolutional residual block: (B, C, H, W) → (B, C, H, W)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class ConvNetworkHead(nn.Module):
    """AlphaZero policy and value heads using 1×1 conv projections."""

    def __init__(self, legal_actions: int, channels: int, height: int, width: int):
        super().__init__()

        # Policy head: 1×1 conv → 2 channels → flatten → linear
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * height * width, legal_actions)

        # Value head: 1×1 conv → 1 channel → flatten → linear → linear
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(height * width, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, action_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))
        if action_mask is not None:
            p = p.masked_fill(~action_mask, float("-inf"))
        p = self.softmax(p)

        v = self.relu(self.value_bn(self.value_conv(x)))
        v = self.relu(self.value_fc1(v.flatten(1)))
        v = self.tanh(self.value_fc2(v))

        return p, v


class MLPEncoder(nn.Module):
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

        self.expected_flat_size = input_shape
        self.leakyrelu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_shape, out_features=hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_shape)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        was_unbatched = (
            observation.dim() > 0 and observation.numel() == self.expected_flat_size
        )
        if was_unbatched:
            observation = observation.unsqueeze(0)

        x = self.flatten(observation)
        x = self.leakyrelu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.leakyrelu(layer(x))
        x = self.output_layer(x)

        if was_unbatched:
            x = x.squeeze(0)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, block_size: int, hidden_dim: int = 128):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU()
        self.layer1 = nn.Linear(block_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.leakyrelu(self.layer1(x))
        activation = self.leakyrelu(self.layer2(activation))
        return self.leakyrelu(activation + x)


class LinearNetworkHead(nn.Module):
    def __init__(self, legal_actions: int, input_shape: int, num_hidden_blocks: int):
        super().__init__()
        self.policy_blocks = nn.ModuleList(
            [ResidualBlock(input_shape) for _ in range(num_hidden_blocks)]
        )
        self.value_blocks = nn.ModuleList(
            [ResidualBlock(input_shape) for _ in range(num_hidden_blocks)]
        )
        self.value_head = nn.Linear(input_shape, 1)
        self.policy_head = nn.Linear(input_shape, legal_actions)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, action_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy_features = x
        for block in self.policy_blocks:
            policy_features = block(policy_features)

        value_features = x
        for block in self.value_blocks:
            value_features = block(value_features)

        value = self.tanh(self.value_head(value_features))
        policy_logits = self.policy_head(policy_features)

        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

        return self.softmax(policy_logits), value
