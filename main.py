from torchrl.data import ReplayBuffer, PrioritizedReplayBuffer
from torchrl.data import LazyMemmapStorage
from torch.optim import AdamW

from src.training.vetle.mcts import MCTS

from src.configuration import Configuration
from src.training.trainer import train
from src.configuration import load_config
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet
from src.utils.record import record_episode
from pettingzoo.classic import tictactoe_v3
from src.environments.environment import build_environment
from tensordict import TensorDict
import torch
import wandb

from accelerate import Accelerator


def generate_training_data(
    replay_buffer: ReplayBuffer, config: Configuration, model=None
) -> ReplayBuffer:  # TODO implement later with MCTS
    env = build_environment("tic_tac_toe")
    env.reset()
    observation, reward, terminated, truncated, info = env.last()
    monte_carlo = MCTS(env=env, config=config)

    for i in range(500):
        policy_values = monte_carlo.run_simulations(1000)
        action = torch.multinomial(policy_values, num_samples=1).item()
        env.step(action)
        observation, reward, terminated, truncated, info = env.last()

        td = TensorDict(
            {
                "observation": observation,
                "value": torch.zeros(1),
                "policies": torch.ones(config.network.legal_actions),
            },
            batch_size=[],
        )
    replay_buffer.add(td)

    return replay_buffer


def training_loop(config: Configuration):
    replay_buffer: ReplayBuffer = PrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        storage=LazyMemmapStorage(max_size=1_000_000),
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = AlphaZeroNet(config.network)

    optimizer = AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.weight_decay,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    for episode in range(config.train.num_episodes):
        replay_buffer = generate_training_data(replay_buffer, config, model)

        if len(replay_buffer) >= config.train.min_replay_size:
            model = train(replay_buffer, model, optimizer, config.train.num_epochs)
            record_episode(config.env_name, episode)

    # TODO implement training loop herefrom src.nn_architecture.network_config import load_config, Configuration


if __name__ == "__main__":
    # Get config
    config = load_config("config.yaml")

    # Initialize wandb
    run = wandb.init(
        entity="deeptactics-arena",
        project="AlphaZero deeptactics",
        config=config.model_dump(),
        mode="disabled",  # disabled offline online
        monitor_gym=True,
    )

    # Start training loop
    training_loop(config)

    run.finish()
