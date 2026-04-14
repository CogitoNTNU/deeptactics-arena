import copy

from pettingzoo import AECEnv
from pettingzoo.classic.chess.chess import raw_env as ChessRawEnv


class CloneableChess(AECEnv):
    """
    Thin wrapper around chess raw_env that adds clone() support for MCTS.
    Uses the raw env directly (no OrderEnforcingWrapper) so state can be copied.
    """

    def __init__(self, render_mode: str | None = None, screen_height: int | None = 800):
        self._render_mode = render_mode
        self._screen_height = screen_height
        self._env = ChessRawEnv(render_mode=render_mode, screen_height=screen_height)
        self._env.reset()

    def clone(self) -> "CloneableChess":
        new = CloneableChess.__new__(CloneableChess)
        new._render_mode = self._render_mode
        new._screen_height = self._screen_height
        src = self._env
        dst = ChessRawEnv(render_mode=self._render_mode, screen_height=self._screen_height)
        dst.reset()

        dst.board = src.board.copy()
        dst.board_history = src.board_history.copy()
        dst.agents = list(src.agents)
        dst.rewards = dict(src.rewards)
        dst._cumulative_rewards = dict(src._cumulative_rewards)
        dst.terminations = dict(src.terminations)
        dst.truncations = dict(src.truncations)
        dst.infos = copy.deepcopy(src.infos)
        dst.agent_selection = src.agent_selection
        dst._agent_selector = copy.deepcopy(src._agent_selector)

        new._env = dst
        return new

    def step(self, action):
        return self._env.step(action)

    def observe(self, agent):
        return self._env.observe(agent)

    def reset(self, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def legal_moves(self):
        from pettingzoo.classic.chess import chess_utils
        return chess_utils.legal_moves(self._env.board)

    @property
    def agent_selection(self):
        return self._env.agent_selection

    @property
    def agents(self):
        return self._env.agents

    @property
    def terminations(self):
        return self._env.terminations

    @property
    def truncations(self):
        return self._env.truncations

    @property
    def rewards(self):
        return self._env.rewards

    @property
    def board(self):
        return self._env.board

    @property
    def board_history(self):
        return self._env.board_history

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    @property
    def metadata(self):
        return self._env.metadata

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def infos(self):
        return self._env.infos

    @property
    def _cumulative_rewards(self):
        return self._env._cumulative_rewards

    def is_done(self):
        return all(self._env.terminations.values()) or all(
            self._env.truncations.values()
        )
