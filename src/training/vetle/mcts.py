import torch

from src.training.vetle.node import Node
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.configuration import Configuration
from gymnasium import Env


class MCTS:
    def __init__(
        self,
        env: Env,
        config: Configuration,
        model: AlphaZeroNet,
        device: torch.device | str = "cpu",
    ):
        self.config = config
        self.device = device
        self.c_puct = self.config.mcts.cpuct
        self.pi_temp = self.config.mcts.pi_temp
        self.exploration_moves = self.config.mcts.exploration_moves
        self.inv_temp = 1 / self.pi_temp

        self.env = env

        self.network = model

        self.root = Node(self.network, env, device=self.device)
        self.root.pred_pol = self.dirichlet(
            self.root.pred_pol, self.root.legal_actions, self.config.mcts.epsilon
        )
        # Initialize root as visited so first simulation expands instead of
        # wasting a rollout on the root's own value estimate.
        self.root.num_visited = 1
        self.root.value = self.root.pred_val.item()
        self.root.avg = self.root.value

    def backpropogate(self, node: Node, value: float) -> None:
        while node.parent is not None:
            node.value += value
            node.num_visited += 1
            node.avg = node.value / node.num_visited

            value = -value
            node = node.parent
        node.value += value
        node.num_visited += 1
        node.avg = node.value / node.num_visited

    def PUCT(self, node: Node) -> int:
        """Calculate PUCT for a node, considering both expanded and unexpanded actions."""
        best_action = None
        best_score = -float("inf")
        sqrt_parent = node.num_visited**0.5

        for action in node.legal_actions:
            prior = float(node.pred_pol[action])
            if action in node.children:
                child = node.children[action]
                Q = -child.avg
                U = self.c_puct * prior * sqrt_parent / (1 + child.num_visited)
            else:
                # Unexpanded actions have Q=0, visit_count=0
                Q = 0.0
                U = self.c_puct * prior * sqrt_parent
            score = Q + U
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def policy(self, node: Node, action) -> float:
        """Calculate pi for given action"""
        val = node.children[action].num_visited ** (self.inv_temp)
        val /= sum(
            [node.children[i].num_visited ** (self.inv_temp) for i in node.children]
        )

        return val

    def dirichlet(self, pred_pol, legal_actions, epsilon):
        alpha = self.config.mcts.dirichlet_alpha
        prior = pred_pol.clone()
        conc = torch.full(
            (len(legal_actions),), alpha, dtype=prior.dtype, device=prior.device
        )

        noise = torch.distributions.Dirichlet(conc).sample()
        prior[legal_actions] = (1 - epsilon) * prior[legal_actions] + epsilon * noise

        prior = prior.clamp_min(0)
        prior = prior / prior.sum()

        return prior

    def traverse(self, node: Node):
        while True:
            if node.terminated or node.truncated or len(node.legal_actions) == 0:
                self.backpropogate(node, node.reward)
                return

            if node.num_visited == 0:
                self.rollout(node)
                return

            action = self.PUCT(node)
            if action not in node.children:
                # Lazy expansion: only create the child we actually need
                child = node.add_child(action, self.network)
                self.rollout(child)
                return

            node = node.children[action]

    def rollout(self, node: Node):
        self.backpropogate(node, node.pred_val.item())

    def run_simulations(self, num_simulations, move_number: int = 0):
        for i in range(num_simulations):
            self.traverse(self.root)

        num_actions = len(self.root.pred_pol)
        a = torch.zeros(num_actions, dtype=torch.float32)

        if move_number < self.exploration_moves:
            for action in self.root.children:
                a[action] = self.policy(self.root, action)
        else:
            # τ → 0: pick the most visited action deterministically
            best_action = max(self.root.children, key=lambda c: self.root.children[c].num_visited)
            a[best_action] = 1.0

        return a
