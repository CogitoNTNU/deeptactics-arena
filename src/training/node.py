from typing import List
from pettingzoo.utils.env import AECEnv


class Node:
    def __init__(self, parent, state, env : AECEnv ):
        self.parent = parent
        self.children: List[Node]
        self.value = 0
        self.policy
        self.visited = 0
        self.state = state
        self.env = env


    def add_child(self, child: "Node"):
        self.children.append(child)