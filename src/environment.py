from src.environments.chess_wrapper import CloneableChess
from src.environments.connect_four_wrapper import CloneableConnectFour
from src.environments.tic_tac_toe_wrapper import CloneableTicTacToe


def build_environment(
    env_name: str,
    render_mode: str | None = None,
) -> CloneableChess | CloneableConnectFour | CloneableTicTacToe:
    match env_name.lower():
        case "chess":
            return CloneableChess(render_mode=render_mode)
        case "connect_four":
            return CloneableConnectFour(render_mode=render_mode)
        case "tic_tac_toe":
            return CloneableTicTacToe(render_mode=render_mode)
        case _:
            raise ValueError(f"Unknown environment: {env_name}")
