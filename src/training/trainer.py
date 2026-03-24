import tempfile

import torch
from torchrl.data import ReplayBuffer
import torch.nn as nn
from tensordict import TensorDict
import wandb

from accelerate import Accelerator

MODELS_PATH = "models"

def train(
    replay_buffer: ReplayBuffer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epochs: int = 10,
    
):
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train(True)

        avg_loss = train_one_epoch(replay_buffer, model, optimizer, accelerator)

        wandb.log({"epoch": epoch, "epoch/loss": avg_loss})

        # TODO: After n iterations, run validation and calculate reward
        # upload model artifact
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_name = f"{MODELS_PATH}/best_model_epoch_{epoch}_loss_{avg_loss:.3f}.pth"
            torch.save(model.state_dict(), model_name)
            artifact = wandb.Artifact("deeptactics_arena", type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)



    return model


def train_one_epoch(
    replay_buffer: list[TensorDict],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    sample_size: int = 16,
) -> float:
    running_loss = 0.0
    sampled_batches = replay_buffer.sample(sample_size)

    for batch in sampled_batches:
        observations = batch["observation"]
        values = batch["value"]
        policies = batch["policies"]

        optimizer.zero_grad()

        pred_policies, pred_values = model.forward(observations)

        loss = loss_function(pred_policies, pred_values, policies, values)
        accelerator.backward(loss)

        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

    wandb.log({"batch/loss": batch_loss})
    return running_loss


def loss_function(
    pred_policies: torch.Tensor,
    pred_values: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    MSE_coeff: float = 1,
) -> float:
    mse = nn.functional.mse_loss(pred_values, values)
    ce = nn.functional.cross_entropy(pred_policies, policies)
    return ce + MSE_coeff * mse
