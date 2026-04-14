from src.training.vetle.mcts import MCTS
from src.training.vetle.node import Node


class FakeNode:
    """Minimal Node stand-in for testing backpropagation."""

    def __init__(self, parent=None):
        self.parent = parent
        self.value = 0.0
        self.num_visited = 0
        self.avg = 0.0


def _make_chain(depth: int) -> list[FakeNode]:
    """Build a parent-chain of FakeNodes (index 0 = leaf, last = root)."""
    nodes = [FakeNode()]
    for _ in range(depth):
        nodes.append(FakeNode())
        nodes[-2].parent = nodes[-1]
    return nodes


def test_backprop_single_value():
    """A single backprop of +1 from a depth-3 leaf should alternate signs up the tree."""
    leaf, mid, root = _make_chain(2)

    MCTS.backpropogate(None, leaf, 1.0)

    # leaf gets +1, mid gets -1, root gets +1
    assert leaf.value == 1.0
    assert leaf.num_visited == 1
    assert leaf.avg == 1.0

    assert mid.value == -1.0
    assert mid.num_visited == 1
    assert mid.avg == -1.0

    assert root.value == 1.0
    assert root.num_visited == 1
    assert root.avg == 1.0


def test_backprop_two_updates():
    """Two backprops with different values accumulate correctly."""
    leaf, mid, root = _make_chain(2)

    MCTS.backpropogate(None, leaf, 1.0)
    MCTS.backpropogate(None, leaf, -0.5)

    # leaf: 1.0 + (-0.5) = 0.5, visited 2
    assert leaf.num_visited == 2
    assert abs(leaf.value - 0.5) < 1e-9
    assert abs(leaf.avg - 0.25) < 1e-9

    # mid: -1.0 + 0.5 = -0.5, visited 2
    assert mid.num_visited == 2
    assert abs(mid.value - (-0.5)) < 1e-9
    assert abs(mid.avg - (-0.25)) < 1e-9

    # root: 1.0 + (-0.5) = 0.5, visited 2
    assert root.num_visited == 2
    assert abs(root.value - 0.5) < 1e-9
    assert abs(root.avg - 0.25) < 1e-9


def test_backprop_root_only():
    """Backprop on a root node (no parent) should update just that node."""
    root = FakeNode()

    MCTS.backpropogate(None, root, 0.7)

    assert root.num_visited == 1
    assert abs(root.value - 0.7) < 1e-9
    assert abs(root.avg - 0.7) < 1e-9


def test_backprop_deep_chain_alternates():
    """In a 5-node chain, signs alternate: +, -, +, -, +."""
    nodes = _make_chain(4)  # indices 0..4, 0=leaf, 4=root

    MCTS.backpropogate(None, nodes[0], 1.0)

    expected_signs = [1.0, -1.0, 1.0, -1.0, 1.0]
    for i, expected in enumerate(expected_signs):
        assert abs(nodes[i].value - expected) < 1e-9, (
            f"Node at depth {i}: expected value {expected}, got {nodes[i].value}"
        )
        assert nodes[i].num_visited == 1
        assert abs(nodes[i].avg - expected) < 1e-9
