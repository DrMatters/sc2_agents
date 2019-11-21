import abc
from typing import Optional, Any
from smac import env


class BaseSCEvaluator(abc.ABC):
    def __init__(self, map_name: Optional[str] = None):
        self.env = env.StarCraft2Env(map_name)

    @abc.abstractmethod
    def evaluate(self, individual) -> Any:
        self.env.reset()
