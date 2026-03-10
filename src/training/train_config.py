from pydantic import BaseModel


class TrainConfiguration(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_episodes: int
    min_replay_size: int
