from src.configuration import load_config
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet
import torch
from torchrl.data.replay_buffers import PrioritizedReplayBuffer
from tensordict import TensorDict
from src.training.trainer import train_one_epoch, loss_function


"""
We want to test:
- loss function gives correct value
- After training a batch, the original parameters have been changed, and check that network is the same
- Check that we can give multiple batches 
"""


def test_loss():
    pred_policies = torch.Tensor([5.0, 5.0, 5.0])
    pred_values = torch.Tensor([2.0, 2.0, 2.0])

    policies = torch.Tensor([3.0, 3.0, 3.0])
    values = torch.Tensor([4.0, 4.0, 4.0])

    correct_value = 13.887510299682617  # torch.nn.functional.mse_loss(pred_values, values) + torch.nn.functional.cross_entropy(pred_policies, policies)

    loss_fn_loss = loss_function(pred_policies, pred_values, policies, values)

    assert loss_fn_loss == correct_value, (
        f"Expected loss of {correct_value}, got {loss_fn_loss}"
    )


def test_batch():
    pass
    observation = torch.zeros((3, 64, 64), dtype=torch.uint8)  # Example observation
    reward = 5.0
    policies = 4.0

    observation_dtype = torch.uint8 if observation.ndim == 3 else torch.float32
    td = TensorDict(
        {
            "observation": torch.as_tensor(
                observation, dtype=observation_dtype, device="cpu"
            ).contiguous(),
            "value": torch.as_tensor(reward, dtype=torch.float32, device="cpu").view(
                ()
            ),
            "policies": torch.as_tensor(
                policies, dtype=torch.float32, device="cpu"
            ).contiguous(),
        },
        batch_size=[],
    )
    replay_buffer = PrioritizedReplayBuffer(alpha=0.6, beta=0.4)
    replay_buffer.add(td)
    network_config = load_config("config.yaml").network
    model: torch.nn.Module = AlphaZeroNet(network_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    params1 = model.parameters()
    train_one_epoch(replay_buffer, model, optimizer)
    params2 = model.parameters()

    for p1, p2 in zip(params1, params2):
        assert not torch.equal(p1, p2), "The training did not change the parameters"


def test_multiple_batches():
    assert True
