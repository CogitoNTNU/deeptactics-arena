import torch

from src.training.vetle.node import Node
from src.nn_architecture.AlphaZeroNet import AlphaZeroNet

from src.configuration import Configuration
from gymnasium import Env

VIRTUAL_LOSS = 1.0


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
        self.eval_batch_size = self.config.mcts.eval_batch_size

        self.env = env
        self.network = model

        self.root = Node(env, device=self.device)
        self._batch_evaluate([self.root])
        self.root.pred_pol = self.dirichlet(
            self.root.pred_pol, self.root.legal_actions, self.config.mcts.epsilon
        )
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
                Q = 0.0
                U = self.c_puct * prior * sqrt_parent
            score = Q + U
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def policy(self, node: Node, action) -> float:
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

    def _select_leaf(self, node: Node) -> tuple[Node, list[Node]]:
        """Traverse to a leaf applying virtual loss along the path."""
        path = []
        while True:
            if node.terminated or node.truncated or len(node.legal_actions) == 0:
                return node, path
            if node.pred_pol is None:
                return node, path

            action = self.PUCT(node)
            if action not in node.children:
                child = node.add_child(action)
            else:
                child = node.children[action]

            child.value += VIRTUAL_LOSS
            child.num_visited += 1
            child.avg = child.value / child.num_visited
            path.append(child)
            node = child

    def _undo_virtual_loss(self, path: list[Node]) -> None:
        for node in path:
            node.value -= VIRTUAL_LOSS
            node.num_visited -= 1
            node.avg = node.value / node.num_visited if node.num_visited > 0 else 0.0

    def _batch_evaluate(self, nodes: list[Node]) -> None:
        to_eval = [
            n for n in nodes
            if n.pred_pol is None and not n.terminated and not n.truncated
        ]
        if not to_eval:
            return

        obs_batch = torch.stack(
            [torch.tensor(n.obs["observation"].copy(), dtype=torch.float32) for n in to_eval]
        ).to(self.device)
        mask_batch = torch.stack(
            [torch.tensor(n.obs["action_mask"], dtype=torch.bool) for n in to_eval]
        ).to(self.device)

        pred_pols, pred_vals = self.network(obs_batch, action_mask=mask_batch)

        for node, pol, val in zip(to_eval, pred_pols, pred_vals):
            node.pred_pol = pol
            node.pred_val = val.squeeze(-1)

    def run_simulations(self, num_simulations: int, move_number: int = 0) -> torch.Tensor:
        sim_done = 0
        while sim_done < num_simulations:
            batch_size = min(self.eval_batch_size, num_simulations - sim_done)

            leaves = []
            paths = []
            for _ in range(batch_size):
                leaf, path = self._select_leaf(self.root)
                leaves.append(leaf)
                paths.append(path)

            self._batch_evaluate(leaves)

            for leaf, path in zip(leaves, paths):
                if leaf.terminated or leaf.truncated or len(leaf.legal_actions) == 0:
                    self.backpropogate(leaf, leaf.reward)
                else:
                    self.backpropogate(leaf, leaf.pred_val.item())
                self._undo_virtual_loss(path)

            sim_done += batch_size

        num_actions = len(self.root.pred_pol)
        a = torch.zeros(num_actions, dtype=torch.float32)

        if move_number < self.exploration_moves:
            for action in self.root.children:
                a[action] = self.policy(self.root, action)
        else:
            best_action = max(
                self.root.children, key=lambda c: self.root.children[c].num_visited
            )
            a[best_action] = 1.0

        return a
