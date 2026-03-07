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
    sampled_batches = replay_buffer.sample(sample_size)
        
    for batch in sampled_batches:
        observations = batch["observation"]
        values = batch["value"]
        policies = batch["policies"]

        optimizer.zero_grad()

        pred_policies, pred_values = model.forward(observations)

        loss = loss_function(pred_policies, pred_values, policies, values)
        loss.backward()

        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

    wandb.log({"batch/loss": batch_loss})
    return running_loss


def loss_function(pred_policies: torch.Tensor, pred_values: torch.Tensor, policies: torch.Tensor, values: torch.Tensor, MSE_coeff: float = 1) ->float:
    mse = nn.functional.mse_loss(pred_values, values)
    ce = nn.functional.cross_entropy(pred_policies, policies)
    return ce + MSE_coeff * mse


