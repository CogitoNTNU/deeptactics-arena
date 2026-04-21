from gymnasium import Env


class Node:
    def __init__(self, env: Env = None, action=None, device: str = "cpu"):
        self.device = device
        self.action = action
        self.env = env.clone()
        if action is not None:
            self.env.step(self.action)
        self.obs, self.reward, self.terminated, self.truncated, _ = self.env.last()

        self.pred_pol = None
        self.pred_val = None

        self.parent: Node = None
        self.children: dict[int, Node] = {}

        self.value: float = 0
        self.num_visited: int = 0
        self.avg: float = 0

        self.legal_actions = [
            i for i in self.env.legal_moves() if self.obs["action_mask"][i]
        ]

    def add_child(self, action: int) -> "Node":
        new_node = Node(self.env, action, device=self.device)
        new_node.parent = self
        self.children[action] = new_node
        return new_node
