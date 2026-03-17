from typing import List
from pettingzoo.utils.env import AECEnv

class Node:
    def __init__(self, parent, state, env: AECEnv):
        self.parent = parent
        self.state = state
        self.env = env
        self.children: List[Node] = []
        self.policy
        self.value = 0
        self.visited = 0

    def add_children(self, child: Node):
        self.children.append(child)
