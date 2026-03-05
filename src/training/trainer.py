import torch
from torchrl.data import ReplayBuffer
import torch.nn as nn
from tensordict import TensorDict
from typing import Callable
import wandb


def train(replay_buffer: ReplayBuffer, model: nn.Module, optimizer: torch.optim.Optimizer, epochs: int = 10):
    for epoch in range(epochs):
        model.train(True)

        avg_loss = train_one_epoch(replay_buffer, model, optimizer)

        wandb.log({"epoch": epoch, "epoch/loss": avg_loss})


def train_one_epoch(replay_buffer: list[TensorDict], model: nn.Module, optimizer: torch.optim.Optimizer, sample_size:int=16) -> float:
    running_loss = 0.0
    batch = replay_buffer.sample(sample_size)
        
    batch_loss = train_per_batch(batch, model, optimizer)
    running_loss += batch_loss
    wandb.log({"batch/loss": batch_loss})
    return running_loss


def train_per_batch(tranjectories: TensorDict, model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    observations = tranjectories["observation"]
    values = tranjectories["value"]
    policies = tranjectories["policies"]

    optimizer.zero_grad()

    pred_policies, pred_values = model(observations)

    loss = loss_function(pred_policies, pred_values, policies, values)
    loss.backward()

    optimizer.step()

    """
    observation_dtype = torch.uint8 if observation.ndim == 3 else torch.float32

        td = TensorDict(
            {
                "observation": torch.as_tensor(
                    observation, dtype=observation_dtype, device="cpu"
                ).contiguous(),
                "value": torch.as_tensor(
                    reward, dtype=torch.float32, device="cpu"
                ).view(()),
                "policies": torch.as_tensor(
                    policies, dtype=torch.float32, device="cpu"
                ).contiguous(),
            
            
            },
            batch_size=[],
        )

        self.replay_buffer.add(td)
    """

    return loss.item()


def loss_function(pred_policies: torch.Tensor, pred_values: torch.Tensor, policies: torch.Tensor, values: torch.Tensor) ->float:
    mse = nn.functional.mse_loss(pred_values, values)
    ce = nn.functional.cross_entropy(pred_policies, policies)
    return ce + mse


