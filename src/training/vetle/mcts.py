import torch

from src.training.vetle.node import Node
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.configuration import Configuration
from copy import deepcopy
from gymnasium import Env

class MCTS():
    def __init__(self, env: Env, config: Configuration):
        self.config = config

        self.env = env
        self.env.reset()

        self.root = Node(env)
        self.root.add_children(env.legal_moves())
        self.network = AlphaZeroNet(config.network)

        
        self.c_puct = self.config.mcts.cpuct
        self.pi_temp = self.config.mcts.pi_temp
        self.inv_temp = 1/self.pi_temp
    
    def backpropogate(self, node: Node, value: float) -> None:
        """Backpropagate value up tree"""
        while node.parent != None:
            node.value += value
            node.num_visited += 1
            node.avg = node.value/node.num_visited

            value = -value
            node = node.parent
    
    def PUCT(self, node: Node) -> float:
        """Calculate PUCT for a node and state"""
        
        pred_pol, pred_val = self.network.forward(torch.tensor(node.obs["observation"], dtype=torch.float32).flatten())

        val = node.avg

        val += self.c_puct * pred_val * node.parent.num_visited**(0.5) / (1+node.num_visited)

        return val
    
    def pi(self, node: Node, action) -> float:
        """Calculate pi for given action"""
        val = node.children[action].num_visited**(self.inv_temp)
        val /= sum([node.children[i].num_visited**(self.inv_temp) for i in node.children])
        
        return val
    

    def dirichlet(self, epsilon, eta):
        
        pred_pol, pred_val = self.network.forward(self.root.obs)
        eta = torch.randn_like(pred_pol)
        prior_prime = (1 - epsilon)*pred_pol + epsilon*eta

        return prior_prime

    

    def traverse(self, node: Node):
        if node.children:
            max_PUCT = -1
            best_node = None
            for child in node.children:
                puct = self.PUCT(node.children[child])
                if puct > max_PUCT:
                    max_PUCT = puct
                    best_node = node.children[child]
            
            self.traverse(best_node)

        elif node.num_visited == 0:
            self.rollout(node)
        else: 
            if self.env.legal_moves():
                mask = node.obs["action_mask"]
                actions = [self.env.legal_moves()[i] for i in range(len(self.env.legal_moves())) if mask[i]]
                print(actions)
                node.add_children(actions)
                self.traverse(node.children[actions[0]])

    def rollout(self, node: Node):
        pred_policy, pred_val = self.network.forward(torch.tensor(node.obs["observation"], dtype=torch.float32).flatten())

        self.backpropogate(node, pred_val)


    def run_simulations(self, num_simulations):
        for i in range(num_simulations):
            self.traverse(self.root)

        a = [lambda x: x.num_visited for i, x in enumerate(self.root.children)]
        #best_action = self.root.children[torch.argmax(a)]

        return a
        

    