from typing import List


class Node:
    def __init__(self, parent, state):
        self.parent = parent
        self.children: List[Node]
        self.value = 0
        self.policy
        self.visited = 0
        self.state = state
