import torch
import numpy as np


class MCTS:
    def __init__(self, state):
        self.root = Node(parent=None, state=state)

    def select(self) -> Node:
        pass

    def expand(self, selected_node) -> torch.Tensor:
        pass

    def backpropagate(self, selected_node, value):
        pass

    def policies(self) -> torch.Tensor:
        pass

    def dirichlet(self):
        pass

    def run_simulations(self, num_simulations) -> torch.Tensor | np.ndArray:
        "select to find out chosen node"
        "expand the selected node"
        "simulate which calls on the network and gets the values"
        "backpropagate to update values up to the root node"

        # TODO add Dirichlet noise

        for i in range(num_simulations):
            selected_node = self.select()
            value = self.expand(selected_node)
            self.backpropagate(selected_node, value)

        return self.policies()
