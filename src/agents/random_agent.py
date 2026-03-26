import random
from numpy import ndarray
from typing import Union
from src.agent import Agent
from torch import Tensor


class RandomAgent(Agent):
    def __init__(self, num_legal_actions: int):
        self.num_legal_actions = num_legal_actions
    
    def act(self, observation: Union[ndarray, Tensor], legal_mask: Union[ndarray, Tensor]) -> int:
        return random.randint(0, self.num_legal_actions - 1)
    
    def load_policy(self, policy_name: str) -> None:
        pass
