from typing import Protocol, Union

from numpy import ndarray
from torch import Tensor


class Agent(Protocol):
    def act(self, observation: Union[ndarray, Tensor], legal_mask: Union[ndarray, Tensor]) -> int:
        """
        Determine the action to take based on the current observation.
        Args:
            observation (Union[ndarray, Tensor]): The current state observation from the environment.
            legal_mask (Union[ndarray, Tensor]): Encodes whether the current state observation is legal.
        Returns:
            int: The action chosen by the agent.
        """
        ...

    def load_policy(self, policy_name: str) -> None:
        """
        Load a saved policy into the agent.

        Args:
            policy_name (str): The name of the policy to load.
        """
        ...
