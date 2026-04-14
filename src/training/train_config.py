from pydantic import BaseModel


class TrainConfiguration(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_episodes: int
    min_replay_size: int
    max_replay_size: int
    num_batches: int
    T_0: int
    num_parallel_games: int = 1
    T_0: int
