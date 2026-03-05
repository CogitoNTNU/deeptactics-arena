import torch
from torchrl.data import ReplayBuffer
import torch.nn as nn
from tensordict import TensorDict
from typing import Callable


def train_per_batch(tranjectories: TensorDict, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable) -> float:
    observations = tranjectories["observation"]
    values = tranjectories["value"]
    policies = tranjectories["policies"]

    optimizer.zero_grad()

    pred_policies, pred_values = model(observations)

    loss = loss_fn(pred_policies, pred_values, policies, values)
    loss.backward()

    optimizer.step()

    """
    observation_dtype = torch.uint8 if observation.ndim == 3 else torch.float32

        td = TensorDict(
            {
                "observation": torch.as_tensor(
                    observation, dtype=observation_dtype, device="cpu"
                ).contiguous(),
                "action": torch.as_tensor(action, dtype=torch.long, device="cpu").view(
                    ()
                ),
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


def loss_with_torch(pred_policies: torch.Tensor, pred_values: torch.Tensor, policies: torch.Tensor, values: torch.Tensor) ->float:
    mse = nn.functional.mse_loss(pred_values, values)

    ce = nn.functional.cross_entropy(pred_policies, policies)
    return ce + mse
