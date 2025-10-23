from dataclasses import dataclass, field
from functools import partial, reduce
from typing import Callable, Any

import jax
from flax import nnx
import jax.numpy as jnp
import pennylane as qml
from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator

# See save/load example: https://github.com/google/flax/blob/main/examples/nnx_toy_examples/08_save_load_checkpoints.py

def module_tagger(cls):
    """Tag serializable modules by their class name."""
    return cls.__name__


@dataclass(kw_only=True)
class SerializableModule(nnx.Module, DataClassDictMixin):
    """Base class for Flax NNX modules that can be serialized to dictionaries.
    
    Variants of this class are distinguished via the `"type"` field that is auto-generated upon serialization. By default, this `"type"` field is the class string name `cls.__name__`. See the `module_tagger` function at module scope.
    """
    class Config(BaseConfig):
        discriminator = Discriminator(
            field="type",
            include_subtypes=True,
            variant_tagger_fn=module_tagger,
        )
    
    @nnx.jit
    def __call__(self, x):
        """Default call function returns the input unchanged."""
        return x
    
    def __pre_serialize__(self):
        return self
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Re-build serialization dictionary by creating a `type` field as the first key.
        d = {
            'type': module_tagger(type(self)), # Discriminator field.
            **d,
        }
        return d