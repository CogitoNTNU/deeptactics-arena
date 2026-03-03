import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.nn_architecture.network_config import NetworkConfig


class AlphaZeroNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super.__init()
        
        encoder_type = config.encoder_type
        match encoder_type:
            case "cnn":
                self.model = CNNEncoder(config.input_shape)
            case "mlp":
                pass
        
        # stem

        # output head



    def forward(self, obs: torch.Tensor):


        return self.model.forward()

    def backward(self):
        pass


class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.module_list = nn.ModuleList([nn.Conv2d()])
    
    def forward(self):
        pass
        
        


class MLPEncoder(nn.Module):    #f : obs -> input
    def __init__(self, num_layers : int, input_shape: int, output_shape : int):
        super().__init__()

        if input_shape <= 0:
            raise ValueError
        if output_shape <= 0:
            raise ValueError

        self.input_layer = nn.Linear(input_shape, out_features=128)
        self.hidden_layers = nn.ModuleList([nn.Linear(128, 128) for i in range(num_layers)])
        self.output_layer = nn.Linear(128, output_shape)



    
    def forward(self, observation : torch.Tensor) -> torch.Tensor:
        x = self.input_layer(observation)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            #x = nn.ReLU(x)
        x = self.output_layer(x)

        return x



