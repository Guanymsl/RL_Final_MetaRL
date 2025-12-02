import abc
from typing import Generic, TypeVar, Tuple

ObsType = TypeVar('ObsType')

class MetaEpisodicEnv(abc.ABC, Generic[ObsType]):
    @property
    @abc.abstractmethod
    def max_episode_len(self) -> int:
        pass

    @abc.abstractmethod
    def new_env(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> ObsType:
        pass

    @abc.abstractmethod
    def step(
        self,
        action: int,
        auto_reset: bool = True
    ) -> Tuple[ObsType, float, bool, dict]:
        pass
