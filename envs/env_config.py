import pydantic
from typing import Callable, Optional
import gymnasium as gym


class EnvConfig(pydantic.BaseModel):
    direct_access: bool = False
    max_steps: int = None

    env_factory: Callable[['EnvConfig'], gym.Env]

    obs_dim: int = None
    a_dim: int = None

    env_name: str = None

    seed: Optional[int] = None
