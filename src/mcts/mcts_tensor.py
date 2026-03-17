import torch
from tensordict import TensorDict
from torchrl.data import TensorStorage

# Change later to use a config file or something
state_shape = (3, 3)  # Example for tic-tac-toe
action_shape = 9,  # 9 possible actions in tic-tac-toe
batch_size = 1  # For simplicity, we can start with a batch size of 1
max_nodes = 100  # Maximum number of nodes in the MCTS tree


device = "cuda" if torch.cuda.is_available() else "cpu"


tree = TensorStorage(
    TensorDict(
        {
            "state": torch.zeros((max_nodes, *state_shape), device=device),
            "action": torch.zeros((max_nodes, *action_shape), device=device),
            "value": torch.zeros(max_nodes, device=device),
            "visit_count": torch.zeros(max_nodes, device=device),
            "parent_index": torch.full((max_nodes,), -1, dtype=torch.long, device=device),
        }
    )
)
