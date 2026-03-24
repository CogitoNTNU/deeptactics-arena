import copy

from pettingzoo import AECEnv
from pettingzoo.classic.connect_four.connect_four import raw_env as ConnectFourRawEnv


class CloneableConnectFour(AECEnv):
    """
    Thin wrapper around connect_four raw_env that adds clone() support for MCTS.
    Uses the raw env directly (no OrderEnforcingWrapper) so state can be deepcopied.
    """

    def __init__(self, render_mode: str | None = None, screen_scaling: int = 9):
        self._render_mode = render_mode
        self._screen_scaling = screen_scaling
        self._env = ConnectFourRawEnv(render_mode=render_mode, screen_scaling=screen_scaling)
        self._env.reset()

    def clone(self) -> "CloneableConnectFour":
        new = CloneableConnectFour.__new__(CloneableConnectFour)
        new._render_mode = self._render_mode
        new._screen_scaling = self._screen_scaling
        src = self._env
        dst = ConnectFourRawEnv(render_mode=self._render_mode, screen_scaling=self._screen_scaling)
        dst.reset()

        dst.board = list(src.board)
        dst.agents = list(src.agents)
        dst.rewards = dict(src.rewards)
        dst._cumulative_rewards = dict(src._cumulative_rewards)
        dst.terminations = dict(src.terminations)
        dst.truncations = dict(src.truncations)
        dst.infos = copy.deepcopy(src.infos)
        dst.agent_selection = src.agent_selection
        # Restore the agent selector to the same position
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
        return self._env._legal_moves()

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
