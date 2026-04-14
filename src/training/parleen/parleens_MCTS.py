from src.training.parleen.parleens_node import Node
from typing import List
from pettingzoo.utils.env import AECEnv
from src.nn_architecture import AlphaZeroNet

class MCTS:
    def __init__(self, state):
        self.root = Node(parent = None, state=state, env = AECEnv())
        self.c = 2.14

    def select(self):
        current_node = self.root

        while current_node.children:
            puct_list = []

            for i in range (len(current_node.children)):
                puct_list.append(self.puct_score(current_node.children[i], current_node.policy[i]))

            best_idx = torch.argmax(puct_list)
            current_node = current_node.children[best_idx]

        return current_node

    def expand(self, selected_node:Node, model:AlphaZeroNet) -> torch.Tensor:
        "Give the state to the NN"
        "Get back the policy list (for the children) and value (which is actually part of sim)"
        "Use the policy list to create the children nodes of the current node"
        "Add the value to the selected node"
        
        policy: torch.Tensor #[legal action 1, legal action 2,..., legal action n]
        value: torch.Tensor #[value]

        policy, value = model.forward(selected_node.state)
        selected_node.policy = policy #The policy is a property of the selected node and not of children

        for i in range (len(selected_node.children)): 
            selected_node.children[i].policy = policy[i]



        

        
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




