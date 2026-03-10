import torch
import numpy as np
from src.training import Node
from src.training import MCTS
from pettingzoo.utils.env import AECEnv

node = Node(parent=None, state=None, env=AECEnv())
MCTS.puct_score(node, policy=None)