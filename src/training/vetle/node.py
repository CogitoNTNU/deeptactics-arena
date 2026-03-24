from gymnasium import Env
from copy import deepcopy

class Node():
    def __init__(self, env: Env = None, action=None):
        self.action = action
        self.env = env.clone()
        if action is not None:
            self.env.step(self.action)
            self.obs, self.reward, self.terminated, self.truncated, self.info = self.env.last()
        else:
            self.env.reset()
            self.obs, self.reward, self.terminated, self.truncated, self.info = self.env.last()

        self.parent: Node = None
        self.children: dict[str, Node] = {}

        self.value: float = 0
        self.num_visited: int = 0
        self.avg: float = 0

    
    def add_children(self, legal_actions):
        for action in legal_actions:
            new_node = Node(self.env, action)
            new_node.parent = self

            self.children[action] = new_node