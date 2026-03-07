from src.training.trainer import *


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

    correct_value = 13.887510299682617 # torch.nn.functional.mse_loss(pred_values, values) + torch.nn.functional.cross_entropy(pred_policies, policies)
    
    loss_fn_loss = loss_function(pred_policies, pred_values, policies, values)

    assert loss_fn_loss == correct_value, f"Expected loss of {correct_value}, got {loss_fn_loss}"

def test_batch():
    observation = 3.0
    reward = 5.0
    policies = 4.0

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
    model: torch.nn.Module = ...
    optimizer = torch.optim.AdamW()
    
    params1 = model.parameters()
    train_one_epoch(td, model, optimizer)
    params2 = model.parameters()

    
    for p1, p2 in zip(params1, params2):
        assert not torch.equal(p1, p2), "The training did not change the parameters"



def test_multiple_batches():
    assert True