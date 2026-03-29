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

        metrics = train_one_epoch(replay_buffer, model, optimizer, config.batch_size, config.num_batches)

        wandb.log({"epoch": epoch, **{f"epoch/{k}": v for k, v in metrics.items()}})

        avg_loss = metrics["loss"]
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
) -> dict:
    device = next(model.parameters()).device
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_value_sign_acc = 0.0

    for _ in range(num_batches):
        batch = replay_buffer.sample(batch_size)

        observations = batch["observation"].to(device)
        values = batch["value"].to(device)
        policies = batch["policies"].to(device)

        optimizer.zero_grad()

        pred_policies, pred_values = model.forward(observations)

        policy_loss, value_loss = loss_components(pred_policies, pred_values, policies, values)
        loss = policy_loss + value_loss
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        optimizer.step()

        entropy = -(pred_policies * torch.log(pred_policies + 1e-8)).sum(dim=-1).mean()
        value_sign_acc = (pred_values.sign() == values.sign()).float().mean()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        total_value_sign_acc += value_sign_acc.item()

        wandb.log({
            "batch/loss": loss.item(),
            "batch/policy_loss": policy_loss.item(),
            "batch/value_loss": value_loss.item(),
            "batch/policy_entropy": entropy.item(),
            "batch/value_sign_accuracy": value_sign_acc.item(),
            "batch/grad_norm": grad_norm.item(),
        })

    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "policy_entropy": total_entropy / num_batches,
        "value_sign_accuracy": total_value_sign_acc / num_batches,
    }


def loss_components(
    pred_policies: torch.Tensor,
    pred_values: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    MSE_coeff: float = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_loss = MSE_coeff * nn.functional.mse_loss(pred_values, values)
    policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8), dim=-1).mean()
    return policy_loss, value_loss


def loss_function(
    pred_policies: torch.Tensor,
    pred_values: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    MSE_coeff: float = 1,
) -> float:
    policy_loss, value_loss = loss_components(pred_policies, pred_values, policies, values, MSE_coeff)
    return policy_loss + value_loss
