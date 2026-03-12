from pettingzoo import AECEnv
from pettingzoo.classic import chess_v6, connect_four_v3, tictactoe_v3


def build_environment(env_name: str) -> AECEnv:
    match env_name.lower():
        case "chess_v6":
            return chess_v6.env()
        case "connect_four_v3":
            return connect_four_v3.env()
        case "tictactoe_v3":
            return tictactoe_v3.env()
        case _:
            raise ValueError(f"Unknown environment: {env_name}")
