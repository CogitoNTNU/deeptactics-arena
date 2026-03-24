from src.tic_tac_toe_wrapper import CloneableTicTacToe


def test_clone_has_same_state():
    """After cloning, the clone should have the same board state."""
    env = CloneableTicTacToe()
    env.step(4)  # player_1 plays center

    clone = env.clone()

    assert clone.board.squares == env.board.squares
    assert clone.agent_selection == env.agent_selection


def test_clone_board_is_different_object():
    """The clone's board and squares list must be distinct objects from the original."""
    env = CloneableTicTacToe()
    env.step(4)

    clone = env.clone()

    assert id(clone.board) != id(env.board)
    assert id(clone.board.squares) != id(env.board.squares)


def test_clone_is_independent():
    """Stepping the clone should not affect the original, and vice versa."""
    env = CloneableTicTacToe()
    env.step(4)  # player_1 plays center

    clone = env.clone()
    squares_before = list(env.board.squares)

    clone.step(0)  # player_2 plays top-left on the clone only

    assert env.board.squares == squares_before, "Original board changed after stepping the clone"
    assert clone.board.squares != squares_before, "Clone board should have changed"
