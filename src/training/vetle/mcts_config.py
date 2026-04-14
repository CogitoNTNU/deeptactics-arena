from pydantic import BaseModel

class MCTSConfiguration(BaseModel):
    cpuct: float
    pi_temp: float
    exploration_moves: int = 30
    epsilon: float
    dirichlet_alpha: float = 0.3
    num_simulations: int = 200