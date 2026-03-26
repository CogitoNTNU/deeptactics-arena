import torch

from src.training.vetle.node import Node
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.configuration import Configuration
from copy import deepcopy
from gymnasium import Env

class MCTS():
    def __init__(self, env: Env, config: Configuration, model):
        self.config = config
        
        self.c_puct = self.config.mcts.cpuct
        self.pi_temp = self.config.mcts.pi_temp
        self.inv_temp = 1/self.pi_temp

        self.env = env
        self.env.reset()
        
        self.network = model

        self.root = Node(self.network, env)
        self.root.pred_pol = self.dirichlet(self.root.pred_pol, self.config.mcts.epsilon)
        self.num_root_actions = self.env.legal_moves()
        
    
    def backpropogate(self, node: Node, value: float) -> None:
        """Backpropagate value up tree"""
        while node.parent != None:
            node.value += value
            node.num_visited += 1
            node.avg = node.value/node.num_visited

            value = -value
            node = node.parent
        node.value += value
        node.num_visited += 1
        node.avg = node.value/node.num_visited
    
    def PUCT(self, node: Node) -> float:
        """Calculate PUCT for a node and state"""
        PUCT_vals = []
        for action in self.num_root_actions:
            if action in node.children:
                val = node.children[action].avg

                val += self.c_puct * node.pred_pol.tolist()[action] * node.num_visited**(0.5) / (1+node.children[action].num_visited)

                PUCT_vals.append(val)
            else:
                PUCT_vals.append(-1e15)

        return torch.argmax(torch.asarray(PUCT_vals)).item()
    
    def policy(self, node: Node, action) -> float:
        """Calculate pi for given action"""
        val = node.children[action].num_visited**(self.inv_temp)
        val /= sum([node.children[i].num_visited**(self.inv_temp) for i in node.children])
        
        return val
    

    def dirichlet(self, pred_pol, epsilon):
        
        eta = torch.randn_like(pred_pol)
        prior_prime = (1 - epsilon)*pred_pol + epsilon*eta

        return prior_prime

    

    def traverse(self, node: Node):
        
        if len(node.children) != 0:
            #print(f"{node.action}: finn neste child {node.children.keys()}")

            best_node = node.children[self.PUCT(node)]
            
            self.traverse(best_node)

        elif node.num_visited == 0:
            #print(f"{node.action}: Ikke besøkt før, gjør rollout")
            self.rollout(node)
        else: 
            
            # mask = node.obs["action_mask"]
            # legal = [node.env.legal_moves()[i] for i in range(len(node.env.legal_moves())) if mask[i]]
            #print(legal, node.terminated, node.truncated)
            legal = node.env.legal_moves()

            if (len(legal)==0) or node.truncated or node.terminated:
                #print(f"{node.action}: Besøkt før: finn mulige actions og gjør en: spillet slutt")
                self.backpropogate(node, node.reward)
            
            else:
                #print(f"{node.action}: Besøkt før: finn mulige actions og gjør en: Legg til barn")
                node.add_children(self.network)
                self.traverse(node.children[legal[0]])

    def rollout(self, node: Node):
        self.backpropogate(node, node.pred_val.item())


    def run_simulations(self, num_simulations):
        for i in range(num_simulations):
            #print(f"{i}----------------------------")
            self.traverse(self.root)
 
        a = torch.asarray([0 if x not in self.root.children else self.policy(self.root, x) for x in self.num_root_actions], dtype=torch.float32)
    
        #print("Ferdig")

        return a
        

    