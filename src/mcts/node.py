class Node:
    def __init__(self, parent, action, state):
        self.parent: Node = parent
        self.children: list[Node] = []
        self.value: float = 0
        self.node_visited: int = 0
        self.action = action
        self.state = state

    def add_children(self, children: list["Node"]):
        self.children.expand(children)
