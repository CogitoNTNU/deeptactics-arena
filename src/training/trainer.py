import torch
from torchrl.data import ReplayBuffer
import torch.nn as nn
from tensordict import TensorDict
from src.training.train_config import TrainConfiguration
import wandb

MODELS_PATH = "models"


def train(
    replay_buffer: ReplayBuffer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfiguration,
):
    best_loss = float("inf")
    for epoch in range(config.num_epochs):
        model.train(True)

        avg_loss = train_one_epoch(replay_buffer, model, optimizer, config.batch_size, config.num_batches)

        wandb.log({"epoch": epoch, "epoch/loss": avg_loss})

        # TODO: After n iterations, run validation and calculate reward
        # upload model artifact
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_name = (
                f"{MODELS_PATH}/best_model_epoch_{epoch}_loss_{avg_loss:.3f}.pt"
            )

            torch.save(model.state_dict(), model_name)
            artifact = wandb.Artifact("deeptactics_arena", type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)

def train_one_epoch(
    replay_buffer: list[TensorDict],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int = 2048,
    num_batches: int = 1000,
) -> float:
    device = next(model.parameters()).device
    total_loss = 0.0

    for _ in range(num_batches):
        batch = replay_buffer.sample(batch_size)

        observations = batch["observation"].to(device)
        values = batch["value"].to(device)
        policies = batch["policies"].to(device)

        optimizer.zero_grad()

        pred_policies, pred_values = model(observations)

        loss = loss_function(pred_policies, pred_values, policies, values)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch/loss": loss.item()})

    return total_loss / num_batches


def loss_function(
    pred_policies: torch.Tensor,
    pred_values: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    MSE_coeff: float = 1,
) -> float:
    mse = nn.functional.mse_loss(pred_values, values)
    cross_entropy = -torch.sum(policies * torch.log(pred_policies + 1e-8), dim=-1).mean()
    return cross_entropy + MSE_coeff * mse
