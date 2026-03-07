import os
from pathlib import Path

from pydantic import BaseModel
import yaml

CONFIG_PATH="./configs"

class StemConfig(BaseModel):
    num_residual_blocks: int
    block_size: int

class HeadConfig(BaseModel):
    hidden_blocks: int


class NetworkConfig(BaseModel):
    encoder_type: str
    input_shape: int | list[int]
    hidden_shape: int
    legal_actions: int
    num_layers: int
    stem: StemConfig
    head: HeadConfig

class Configuration(BaseModel):
    network: NetworkConfig
    env_name: str

def load_config(path:str | Path)->Configuration:
    path = os.path.join(CONFIG_PATH, path)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Configuration(**raw_config)

