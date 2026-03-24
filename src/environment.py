from src.environments.chess_wrapper import CloneableChess
from src.environments.connect_four_wrapper import CloneableConnectFour
from src.environments.tic_tac_toe_wrapper import CloneableTicTacToe


def build_environment(
    env_name: str,
) -> CloneableChess | CloneableConnectFour | CloneableTicTacToe:
    match env_name.lower():
        case "chess":
            return CloneableChess()
        case "connect_four":
            return CloneableConnectFour()
        case "tic_tac_toe":
            return CloneableTicTacToe()
        case _:
            raise ValueError(f"Unknown environment: {env_name}")
