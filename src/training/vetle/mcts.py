import torch

from src.training.vetle.node import Node
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.configuration import Configuration
from copy import deepcopy
from gymnasium import Env


class MCTS:
    def __init__(
        self, env: Env, config: Configuration, model, device: torch.device = "cpu"
    ):
        self.config = config
        self.device = device
        self.c_puct = self.config.mcts.cpuct
        self.pi_temp = self.config.mcts.pi_temp
        self.inv_temp = 1 / self.pi_temp

        self.env = env
        self.env.reset()

        self.network = model

        self.root = Node(self.network, env)
        self.root.pred_pol = self.dirichlet(self.root.pred_pol,
                                            self.root.legal_actions,
                                            self.config.mcts.epsilon)
        self.num_root_actions = self.env.legal_moves()

    def backpropogate(self, node: Node, value: float) -> None:
        """Backpropagate value up tree"""
        while node.parent != None:
            node.value += value
            node.num_visited += 1
            node.avg = node.value / node.num_visited

            value = -value
            node = node.parent
        node.value += value
        node.num_visited += 1
        node.avg = node.value / node.num_visited

    def PUCT(self, node: Node) -> float:
        """Calculate PUCT for a node and state"""
        actions = list(node.children.keys())
        puct_vals = []
        for action in actions:
            child = node.children[action]
            Q = child.avg
            U = self.c_puct * float(node.pred_pol[action]) * (node.num_visited ** 0.5) / (1 + child.num_visited)
            puct_vals.append(Q+U)

        best_idx = int(torch.argmax(torch.tensor(puct_vals)))
        return actions[best_idx]
    

    def policy(self, node: Node, action) -> float:
        """Calculate pi for given action"""
        val = node.children[action].num_visited ** (self.inv_temp)
        val /= sum(
            [node.children[i].num_visited ** (self.inv_temp) for i in node.children]
        )

        return val

    def dirichlet(self, pred_pol, legal_actions, epsilon, alpha=0.3):
        prior = pred_pol.clone()
        conc = torch.full(
            (len(legal_actions),), alpha, dtype=prior.dtype, device=prior.device
        )

        if self.device.type == "mps":
            noise = (
                torch.distributions.Dirichlet(conc.to("cpu")).sample().to(self.device)
            )
        else:
            noise = torch.distributions.Dirichlet(conc).sample()
        # på bare legal actions
        prior[legal_actions] = (1 - epsilon) * prior[legal_actions] + epsilon * noise

        # normaliserer
        prior = prior.clamp_min(0)
        prior = prior / prior.sum()

        return prior

    def traverse(self, node: Node):

        if len(node.children) != 0:
            # print(f"{node.action}: finn neste child {node.children.keys()}")

            best_node = node.children[self.PUCT(node)]

            self.traverse(best_node)

        elif node.num_visited == 0:
            # print(f"{node.action}: Ikke besøkt før, gjør rollout")
            self.rollout(node)
        else:

            # mask = node.obs["action_mask"]
            # legal = [node.env.legal_moves()[i] for i in range(len(node.env.legal_moves())) if mask[i]]
            # print(legal, node.terminated, node.truncated)
            legal = node.env.legal_moves()

            if (len(legal) == 0) or node.truncated or node.terminated:
                # print(f"{node.action}: Besøkt før: finn mulige actions og gjør en: spillet slutt")
                self.backpropogate(node, node.reward)

            else:
                # print(f"{node.action}: Besøkt før: finn mulige actions og gjør en: Legg til barn")
                node.add_children(self.network)
                best_action = self.PUCT(node)
                self.traverse(node.children[best_action])

    def rollout(self, node: Node):
        self.backpropogate(node, node.pred_val.item())

    def run_simulations(self, num_simulations):
        for i in range(num_simulations):
            # print(f"{i}----------------------------")
            self.traverse(self.root)

        a = torch.asarray(
            [
                0 if x not in self.root.children else self.policy(self.root, x)
                for x in self.num_root_actions
            ],
            dtype=torch.float32,
        )

        # print("Ferdig")

        return a
