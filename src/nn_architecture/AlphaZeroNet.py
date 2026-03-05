import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.nn_architecture.network_config import NetworkConfig


class AlphaZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init()
        
        encoder_type = config.encoder_type
        match encoder_type:
            case "cnn":
                self.model = CNNEncoder(config.input_shape)
            case "mlp":
                pass
        
        # stem
        self.common_block = nn.ModuleList([ResidualBlock(config.stem.hidden_dim, config.stem.hidden_dim) for i in config.stem.num_residual_blocks])
        
        # output head
        self.head = NetworkHead(config.legal_actions, config.net.hidden_dim, config.head.hidden_blocks)
        

    def forward(self, obs: torch.Tensor):
        x = self.model.forward(obs)
        x = self.common_block.forward(x)
        
        # pred of values and policies
        policies, values = self.head.forward(x)
        return policies, values

    def backward(self):
        pass


class CNNEncoder(nn.Module):
    def __init__(self, num_layers: int, input_shape: int, output_shape: int):     #TODO Endre til tuple input osv

        super().__init__()

        if input_shape <=0:
            raise ValueError

        if output_shape <=0:
            raise ValueError
        
        self.module_list = nn.ModuleList([nn.Conv2d()])
    
    def forward(self):
        pass
        
        


class MLPEncoder(nn.Module):    #f : obs -> input
    def __init__(self, num_layers: int, input_shape: int, output_shape: int):
        super().__init__()

        if input_shape <= 0:
            raise ValueError
        if output_shape <= 0:
            raise ValueError

        self.input_layer = nn.Linear(input_shape, out_features=128)
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for i in range(num_layers)])
        self.output_layer = nn.Linear(128, output_shape)

    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(observation)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = nn.ReLU()(x)
        x = self.output_layer(x)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dim: int=128):
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.layer1(x)
        activation = nn.LeakyReLU(activation)
        activation = self.layer2(x)
        activation = nn.LeakyReLU(activation)
        
        activation += x
        activation = nn.LeakyReLU(activation)
        
        return activation
    

class NetworkHead(nn.modules):
    def __init__(self, legal_actions, input_shape, num_hidden_blocks):
        self.common_block = nn.ModuleList([ResidualBlock(input_shape, input_shape) for i in num_hidden_blocks])

        self.value_head = nn.Linear(input_shape,1)
        self.policy_head = nn.Linear(input_shape, legal_actions)
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        value = self.value_head(x)
        value = nn.Tanh(value)
        
        policy_logits = self.policy_head(x)
        policy_logits = nn.Softmax(policy_logits)

        # [B, A], [B, 1]
        return policy_logits, value
        