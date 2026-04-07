from src.training.kristian.node import Node
from src.nn_architecture import AlphaZeroNet
import numpy as np
import torch
from pettingzoo import AECEnv
from numpy import sqrt


class Mcts:
    def __init__(self, env : AECEnv, model : AlphaZeroNet, c_puct : float):
        self.model = model
        self.env = env
        self.c_puct = float(c_puct)

    
    def get_puct(self,node,child):
        q_value = child.get_mean_value()
        u_value = self.c_puct * child.prior * (sqrt(max(1, (node.visit_count))) / (1+ child.visit_count))
        return q_value + u_value
    
    def select(self, node : Node) -> Node:
        #finner child med høyest puct score:
        current = node

        while current.is_expanded() and not current.is_terminal:
            best_child = None
            best_score = float('-inf')
            for action, child in current.children.items():
                score = self.get_puct(current,child)
                if score>best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                break

            current = best_child

        return current
    
    #TODO: må få inn legal_moves og self.model(state.unsqueeze(0))

    def expand(self, state, node: Node):
        with torch.no_grad():
            netpolicy, value = self.model(state.unsqueeze(0))
        netpolicy = netpolicy.squeeze(0)

        legal_actions = node.env.legal_moves()
        policy_mask = torch.zeros_like(netpolicy)
        policy_mask[legal_actions] = netpolicy[legal_actions]
        if torch.sum(policy_mask) > 0:
            policy_mask = policy_mask/ (torch.sum(policy_mask))
        else:
            policy_mask[legal_actions] = 1.0 / len(legal_actions)

        for action in legal_actions:
            child_env = node.env.clone()
            child_env.step(action)

            child_obs, _ , terminated, truncated, _ = child_env.last()
            child_terminal = bool(terminated or truncated)

            child = Node(
                state = child_obs,
                parent = node,
                action_from_parent=action,
                prior = float(policy_mask[action]),
                is_terminal=child_terminal,
                env=child_env
            )
            node.add_child(action,child)

        return float(value.squeeze().item())
    
    def backprop(self, node:Node, value:float) -> None:
        current = node
        v = float(value)
        while current is not None:
            current.visit_count += 1
            current.value_sum +=v
            v = -v #kun for two player spill
            current = current.parent

    def add_dirichlet_noise(self,root:Node, alpha :float =0.3, epsilon: float = 0.25) ->None:
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))

        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1.0-epsilon) * child.prior + epsilon * float(noise[i])

        
        
    def simulate(self, root:Node) -> float:
        leaf = self.select(root)
        val = self.expand(leaf.state,leaf)

        self.backprop(leaf,val)
        return val

    def run_simulation(self,root: Node, num_simulations : int) -> None:
        if not root.is_expanded():
            self.expand(root.state,root)

        self.add_dirichlet_noise(root)
        
        for _ in range(num_simulations):
            self.simulate(root)

            
        
    
    
