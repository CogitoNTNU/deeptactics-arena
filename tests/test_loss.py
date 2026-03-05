import torch

"""
def test_loss():
    ""
    Test if built-in function and own implementation returns the same value
    ""
    policies: torch.Tensor = torch.rand(10)
    pred_policies: torch.Tensor = torch.rand(10)
    values: torch.Tensor = torch.rand(10)
    pred_values: torch.Tensor = torch.rand(10)

    loss1 = loss_with_torch(pred_policies, pred_values, policies, values)
    loss2 = loss_function(pred_policies, pred_values, policies, values)
    print(loss1)
    print(loss2)
    assert  torch.allclose(loss1, loss2) , "Our implementation does not give same loss as built-in function"
"""