from gymnasium import Env
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Node:
    def __init__(self, model, env: Env = None, action=None):
        self.action = action
        self.env = env.clone()
        if action is not None:
            self.env.step(self.action)
            self.obs, self.reward, self.terminated, self.truncated, _ = self.env.last()
        else:
            self.env.reset()
            self.obs, self.reward, self.terminated, self.truncated, _ = self.env.last()

        self.pred_pol, self.pred_val = model.forward(
            torch.tensor(self.obs["observation"], dtype=torch.float32).to(device)
        )

        self.parent: Node = None
        self.children: dict[str, Node] = {}

        self.value: float = 0
        self.num_visited: int = 0
        self.avg: float = 0

        self.legal_actions = [
            i for i in self.env.legal_moves() if self.obs["action_mask"][i]
        ]

        self.policies = [0 for i in range(len(self.env.legal_moves()))]

    def add_children(self, model):
        for action in self.legal_actions:
            new_node = Node(model, self.env, action)
            new_node.parent = self

            self.children[action] = new_node
