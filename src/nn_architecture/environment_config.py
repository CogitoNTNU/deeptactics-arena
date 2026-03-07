from pydantic import BaseModel

class EnvironmentConfig(BaseModel):
    env_name: str
    seed: int = 42
    render_mode: str

