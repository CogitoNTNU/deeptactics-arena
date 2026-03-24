from src.environments.chess_wrapper import CloneableChess


def test_clone_has_same_state():
    """After cloning, the clone should have the same board state."""
    env = CloneableChess()
    white_move = env.legal_moves()[0]
    env.step(white_move)

    clone = env.clone()

    assert clone.board.fen() == env.board.fen()
    assert clone.agent_selection == env.agent_selection


def test_clone_board_is_different_object():
    """The clone's board must be a distinct object from the original."""
    env = CloneableChess()
    env.step(env.legal_moves()[0])

    clone = env.clone()

    assert id(clone.board) != id(env.board)
    assert id(clone.board_history) != id(env.board_history)


def test_clone_is_independent():
    """Stepping the clone should not affect the original, and vice versa."""
    env = CloneableChess()
    env.step(env.legal_moves()[0])  # white moves

    clone = env.clone()
    fen_before = env.board.fen()

    clone.step(clone.legal_moves()[0])  # black moves on the clone only

    assert env.board.fen() == fen_before, (
        "Original board changed after stepping the clone"
    )
    assert clone.board.fen() != fen_before, "Clone board should have changed"
