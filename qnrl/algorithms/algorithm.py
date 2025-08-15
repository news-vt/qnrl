from dataclasses import dataclass, field
import enum
from typing import Any

from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator

from ..gym import *


def algorithm_tagger(cls):
    """Tag serializable algorithms by their class name."""
    return cls.__name__


class AlgorithmIdentifier(enum.StrEnum):
    """Identifiers for each algorithm."""
    DQN = enum.auto()
    C51 = enum.auto()


@dataclass(kw_only=True)
class Algorithm(DataClassDictMixin):
    """Configuration for one of the `qnrl.algorithm.*.train` methods.
    
    This class contains a `run()` method, that is intended to invoke the `qnrl.algorithm.*.train()` function, along with any other training procedures. The `run()` method will be called with no arguments.
    """
    gym: GymConfig # Gymnasium environment configuration.
    algorithm: AlgorithmIdentifier = field(init=False)
    
    def run(self) -> dict:
        raise NotImplementedError
    
    class Config(BaseConfig):
        discriminator = Discriminator(
            field="type",
            include_subtypes=True,
            variant_tagger_fn=algorithm_tagger,
        )
    
    def __pre_serialize__(self):
        return self
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Re-build serialization dictionary by creating a `type` field as the first key.
        d = {
            'type': algorithm_tagger(type(self)), # Discriminator field.
            **d,
        }
        return d