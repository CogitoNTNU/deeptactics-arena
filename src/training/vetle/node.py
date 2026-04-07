from gymnasium import Env
from copy import deepcopy
import torch


class Node:
    def __init__(self, model, env: Env = None, action=None, device: str = "cpu"):
        self.device = device
        self.action = action
        self.env = env.clone()
        if action is not None:
            self.env.step(self.action)
        self.obs, self.reward, self.terminated, self.truncated, _ = self.env.last()

        obs_tensor = torch.tensor(self.obs["observation"], dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(self.obs["action_mask"], dtype=torch.bool).to(self.device)
        self.pred_pol, self.pred_val = model.forward(obs_tensor, action_mask=mask_tensor)

        self.parent: Node = None
        self.children: dict[int, Node] = {}

        self.value: float = 0
        self.num_visited: int = 0
        self.avg: float = 0

        self.legal_actions = [
            i for i in self.env.legal_moves() if self.obs["action_mask"][i]
        ]

    def add_children(self, model):
        for action in self.legal_actions:
            new_node = Node(model, self.env, action, device=self.device)
            new_node.parent = self

            self.children[action] = new_node
