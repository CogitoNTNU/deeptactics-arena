import os
from pathlib import Path

from pydantic import BaseModel
import yaml

from src.nn_architecture.network_config import NetworkConfig
from src.training.train_config import TrainConfiguration
from src.training.vetle.mcts_config import MCTSConfiguration

CONFIG_PATH = "./configs"


class Configuration(BaseModel):
    network: NetworkConfig
    train: TrainConfiguration
    mcts: MCTSConfiguration
    env_name: str
    weight_decay: float = 1e-4


def load_config(path: str | Path) -> Configuration:
    path = os.path.join(CONFIG_PATH, path)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Configuration(**raw_config)
