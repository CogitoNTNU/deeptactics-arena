from pydantic import BaseModel

class Configuration(BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
