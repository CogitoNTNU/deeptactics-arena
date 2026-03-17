from src.training.parleen.parleens_node import Node
from typing import List
from pettingzoo.utils.env import AECEnv

class MCTS:
    def __init__(self, state):
        self.root = Node(parent = None, state=state, env = AECEnv())
        self.c = 2.14

    def select():
        current_node = self.root

        while current_node.children:
            puct_list = []

            for i in range (len(current_node.children)):
                puct_list.append(self.puct_score(current_node.children[i], current_node.policy[i]))

            best_idx = torch.argmax(puct_list)
            current_node = current_node.children[best_idx]

        return current_node

    def expand():
        "Give the state to the NN"
        "Get back the policy list (for the children) and value (which is actually part of sim)"
        
        


        pass

    def run_simulations():
        pass

    def backpropagate():
        pass

    def puct_score(self, selected_node, policy):
        pass

    def policies():
        pass

    def dirichlet():
        pass




