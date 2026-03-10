import numpy as np
import pyspiel
from node import Node

INF = 10000000


class MCTS:
    def __init__(self, c=1.41):
        """
        Args:
            c (float): The exploration parameter. Higher values encourage exploration, while lower values encourage exploitation.
        """
        self.c = c

    def ucb(self, node: Node):
        """""" """"""
        if node.node_visited == 0:
            return INF
        exploitation = node.value / node.node_visited
        exploration = np.sqrt(np.log(node.parent.node_visited) / node.node_visited)

        return exploitation + self.c * exploration

    def select(self, node: Node):
        """
        Select stage of MCTS.
        Go through the game tree, layer by layer.
        Chooses the node with the highest UCB-score at each layer.
        Returns the selected leaf node.
        """
        if len(node.children) == 0:
            return node

        return self.select(max(node.children, key=lambda child: self.ucb(child)))

    def expand(self, node: Node):
        """
        Optional stage in the MCTS algorithm.
        If you select a leaf node, this method will not be run.

        You expand once per node, you expand by adding all possible children to the children list.
        """
        legal_actions = node.state.legal_actions()
        for action in legal_actions:
            next_state, _, _, _, _ = node.state.step(action)
            child_node = Node(node, action, next_state)
            node.add_child(child_node)

    def rollout(self, node: Node, game: Game):
        """
        Simulate random moves until you reach a leaf node (A conclusion of the game)
        """
        state = node.state.clone()
        children = node.children
        if not children:
            return
        pass

    def backpropagate(self, node: Node, result: int):
        """
        Return the results all the way back up the game tree.
        """

        if node.parent is None:
            return

        node.node_visited += 1
        node.value += result
        self.backpropagate(node.parent, result)

    def run_simulations(self, state, num_simulations=1_000):
        """
        Simulate a game to its conclusion.
        Random moves are selected all the way.
        """

        for n in range(num_simulations):
            selected_node = self.select(state)
            if selected_node.node_visited == 0:
                self.rollout(selected_node, game)
                self.backpropagate(selected_node, selected_node.value)
            else:
                self.expand(selected_node)
                child_node = selected_node.children[0]
                self.rollout(child_node)
                self.backpropagate(child_node, child_node.value)


if __name__ == "__main__":
    game = pyspiel.load_game("tic_tac_toe")
    mcts = MCTS()
    state = game.new_initial_state()

    while not state.is_terminal():
        action = mcts.run_simulations(state, 1_000)
        print("best action\t", action, "\n")
        state.apply_action(action)
        print(state)
        print()
