from src.connect_four_wrapper import CloneableConnectFour


def test_clone_has_same_state():
    """After cloning, the clone should have the same board state."""
    env = CloneableConnectFour()
    env.step(3)  # player_0 drops in column 3

    clone = env.clone()

    assert clone.board == env.board
    assert clone.agent_selection == env.agent_selection


def test_clone_is_independent():
    """Stepping the clone should not affect the original, and vice versa."""
    env = CloneableConnectFour()
    env.step(3)  # player_0 drops in column 3

    clone = env.clone()
    board_before = list(env.board)

    clone.step(4)  # player_1 drops in column 4 on the clone only

    assert env.board == board_before, "Original board changed after stepping the clone"
    assert clone.board != board_before, "Clone board should have changed"
