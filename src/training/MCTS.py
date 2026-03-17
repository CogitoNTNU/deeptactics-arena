from src.nn_architecture import AlphaZeroNet
import torch
import numpy as np
from src.training import Node
from pettingzoo.utils.env import AECEnv

class MCTS:
    def __init__(self, state):
        self.root = Node(parent=None, state=state)
        self.c = 2.14

    def puct_score(self, selected_node, policy):
        policy_int = policy.item()
        
        q = selected_node.value/selected_node.visited
        u = self.c * policy_int * (np.root(selected_node.visited)/(1 + selected_node.visited))
        puct = q + u
        return puct
        

    def select(self) -> Node:
        current_node = self.root

        while current_node.children: 
            puct_list = []
            
            for i in range(len(current_node.children)):
                puct_list.append(self.puct_score(current_node.children[i], current_node.policy[i]))
            
            best_idx = torch.argmax(puct_list)
            current_node = current_node.children[best_idx]
        
        return current_node
            


    def expand(self, selected_node:Node, model: AlphaZeroNet) -> torch.Tensor:
        policy: torch.Tensor# [B, legal actions]
        value: torch.Tensor # [B,1]
        old_policy = selected_node.policy

        #new policy
        policy, value = model.forward(selected_node.state)
        selected_node.policy = policy

        #TODO
        # copy game for each legal action
        #for each legal action, step game with action
        #for each observation, create new node with correct env and state
        selected_node.env.step()["action.mask"]
        observation, reward, termination, truncation, info = env.last()

       
        
        #child list
        child_list = []
        for item in action_list:    #får ut action list fra .step()
                item.add_child(selected_node, )   #lag child for hver action i action list
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
