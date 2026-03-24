import copy

from pettingzoo.classic.tictactoe.tictactoe import raw_env as TicTacToeRawEnv


class CloneableTicTacToe:
    """
    Thin wrapper around tictactoe raw_env that adds clone() support for MCTS.
    Uses the raw env directly (no OrderEnforcingWrapper) so state can be copied.
    """

    def __init__(self, render_mode: str | None = None, screen_height: int | None = 1000):
        self._render_mode = render_mode
        self._screen_height = screen_height
        self._env = TicTacToeRawEnv(render_mode=render_mode, screen_height=screen_height)
        self._env.reset()

    def clone(self) -> "CloneableTicTacToe":
        new = CloneableTicTacToe.__new__(CloneableTicTacToe)
        new._render_mode = self._render_mode
        new._screen_height = self._screen_height
        src = self._env
        dst = TicTacToeRawEnv(render_mode=self._render_mode, screen_height=self._screen_height)
        dst.reset()

        dst.board.squares = list(src.board.squares)
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

    def legal_moves(self):
        return self._env.board.legal_moves()

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

    def is_done(self):
        return all(self._env.terminations.values()) or all(
            self._env.truncations.values()
        )
