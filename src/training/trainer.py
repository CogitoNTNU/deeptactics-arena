import torch
from torchrl.data import ReplayBuffer, TensorDictPrioritizedReplayBuffer
import torch.nn as nn
from src.training.train_config import TrainConfiguration
import wandb

MODELS_PATH = "models"


def train(
    replay_buffer: TensorDictPrioritizedReplayBuffer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfiguration,
):
    best_loss = float("inf")
    for epoch in range(config.num_epochs):
        model.train(True)

        metrics = train_one_epoch(
            replay_buffer, model, optimizer, config.batch_size, config.num_batches
        )

        wandb.log(
            {
                "epoch": epoch,
                "epoch/loss": metrics["loss"],
                "epoch/policy_loss": metrics["policy_loss"],
                "epoch/value_loss": metrics["value_loss"],
                "epoch/policy_entropy": metrics["policy_entropy"],
                "epoch/policy_accuracy": metrics["policy_accuracy"],
                "epoch/value_sign_accuracy": metrics["value_sign_accuracy"],
                "epoch/value_mae": metrics["value_mae"],
            }
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            model_name = (
                f"{MODELS_PATH}/best_model_epoch_{epoch}_loss_{metrics['loss']:.3f}.pt"
            )

            torch.save(model.state_dict(), model_name)
            artifact = wandb.Artifact("deeptactics_arena", type="model")
            artifact.add_file(model_name)
            wandb.log_artifact(artifact)


def train_one_epoch(
    replay_buffer: TensorDictPrioritizedReplayBuffer,
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
    total_policy_acc = 0.0
    total_value_sign_acc = 0.0
    total_value_mae = 0.0

    for _ in range(num_batches):
        batch = replay_buffer.sample()

        observations = batch["observation"].to(device)
        action_masks = batch["action_mask"].to(device)
        values = batch["value"].to(device)
        policies = batch["policies"].to(device)

        optimizer.zero_grad()

        pred_policies, pred_values = model.forward(
            observations, action_mask=action_masks
        )

        policy_loss, value_loss, loss, per_batch_loss = loss_function(
            pred_policies, pred_values, policies, values
        )

        batch.set("td_error", per_batch_loss.abs().detach().cpu())
        replay_buffer.update_tensordict_priority(batch)

        loss.backward()

        grad_norm = (
            sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None)
            ** 0.5
        )

        optimizer.step()

        with torch.no_grad():
            entropy = -(pred_policies * (pred_policies + 1e-8).log()).sum(dim=-1).mean()
            pred_vals_flat = pred_values.squeeze(-1)
            value_sign_acc = (pred_vals_flat.sign() == values.sign()).float().mean()
            value_mae = (pred_vals_flat - values).abs().mean()
            policy_acc = (
                (pred_policies.argmax(-1) == policies.argmax(-1)).float().mean()
            )

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        total_policy_acc += policy_acc.item()
        total_value_sign_acc += value_sign_acc.item()
        total_value_mae += value_mae.item()

        wandb.log(
            {
                "batch/loss": loss.item(),
                "batch/policy_loss": policy_loss.item(),
                "batch/value_loss": value_loss.item(),
                "batch/policy_entropy": entropy.item(),
                "batch/policy_accuracy": policy_acc.item(),
                "batch/value_sign_accuracy": value_sign_acc.item(),
                "batch/value_mae": value_mae.item(),
                "batch/grad_norm": grad_norm.item(),
            }
        )

    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "policy_entropy": total_entropy / num_batches,
        "policy_accuracy": total_policy_acc / num_batches,
        "value_sign_accuracy": total_value_sign_acc / num_batches,
        "value_mae": total_value_mae / num_batches,
    }


def loss_function(
    pred_policies: torch.Tensor,
    pred_values: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    MSE_coeff: float = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (cross_entropy, mse, total, total_per_sample).

    The first three are reduced scalars; total_per_sample has shape (batch,)
    for use with Prioritized Experience Replay priority updates.
    """
    mse_per_sample = nn.functional.mse_loss(
        pred_values, values.unsqueeze(-1), reduction="none"
    ).squeeze(-1)
    ce_per_sample = -torch.sum(policies * torch.log(pred_policies + 1e-8), dim=-1)
    total_per_sample = ce_per_sample + MSE_coeff * mse_per_sample
    return (
        ce_per_sample.mean(),
        mse_per_sample.mean(),
        total_per_sample.mean(),
        total_per_sample,
    )
