from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data import LazyMemmapStorage
from torch import nn
from src.configuration import Configuration
from src.training.trainer import train
from torch.optim import AdamW
from src.nn_architecture.network_config import load_config


def generate_training_data(replay_buffer: ReplayBuffer, model=None) -> ReplayBuffer: #TODO implement later with MCTS
    return replay_buffer


def training_loop(num_episodes: int):
    replay_buffer: ReplayBuffer = PrioritizedReplayBuffer(
        alpha=0.7, 
        beta=0.9,
        storage = LazyMemmapStorage(size=1_000_000),
    )
    model: nn.Module = None
    config = load_config("config.yaml")
    

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)


    for episode in range(num_episodes):
        training_data = generate_training_data(replay_buffer, model)

        if len(replay_buffer) <= config.min_replay_size:
            train(replay_buffer, model, optimizer, config.epochs)
            
    #TODO implement training loop herefrom src.nn_architecture.network_config import load_config, Configuration
