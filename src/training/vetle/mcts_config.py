from pydantic import BaseModel

class MCTSConfiguration(BaseModel):
    cpuct: float
    pi_temp: float
    epsilon: float