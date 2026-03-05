from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

class StemConfig(BaseModel):
    num_residual_blocks: int

class HeadConfig(BaseModel):
    hidden_blocks: int


class NetworkConfig(BaseModel):
    encoder_type: str
    input_shape: int | tuple[int]
    hidden_shape: int
    output_shape: int
    stem: StemConfig
    head: HeadConfig



"""
a = NetworkConfig(layers=1)

to_yaml_file("output/config.yml",a)

c = parse_yaml_file_as(NetworkConfig, "output/config.yml")

print(c)
"""