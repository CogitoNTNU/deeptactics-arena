from pydantic import BaseModel

class StemConfig(BaseModel):
    num_residual_blocks: int
    block_size: int

class HeadConfig(BaseModel):
    hidden_blocks: int

class NetworkConfig(BaseModel):
    encoder_type: str
    input_shape: int | list[int]
    hidden_shape: int
    legal_actions: int
    num_layers: int
    stem: StemConfig
    head: HeadConfig

