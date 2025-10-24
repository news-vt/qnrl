from dataclasses import dataclass, field
import enum
from typing import Any

from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator

from ..checkpoint import CheckpointConfig
from ..metric import MetricManager
from ..gym import EnvConfig, EnvWrapperConfig


def algorithm_tagger(cls):
    """Tag serializable algorithms by their class name."""
    return cls.__name__


@dataclass(kw_only=True)
class Algorithm(DataClassDictMixin):
    """Configuration for one of the `qnrl.algorithm.*.train` methods.
    
    This class contains a `run()` method, that is intended to invoke the `qnrl.algorithm.*.train()` function, along with any other training procedures. The `run()` method will be called with no arguments.
    """
    env: EnvConfig | EnvWrapperConfig # Gymnasium environment configuration.
    seed: int = 0 # Random seed.
    metric_manager: MetricManager | None = None # Metric management configuration.
    checkpoint: CheckpointConfig | None = None  # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.
    
    def train(self, *args, **kwargs) -> list:
        raise NotImplementedError

    def eval(self, *args, **kwargs) -> list:
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