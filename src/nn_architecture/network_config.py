from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

class NetworkConfig(BaseModel):
    encoder_type: str
    input_shape: int | tuple[int]
    hidden_shape: int
    output_shape: int


"""
a = NetworkConfig(layers=1)

to_yaml_file("output/config.yml",a)

c = parse_yaml_file_as(NetworkConfig, "output/config.yml")

print(c)
"""