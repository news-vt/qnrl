from dataclasses import dataclass, field
import enum
from typing import Optional, Union, Any, TypeAlias

import jax
from flax import nnx
import optax
from mashumaro.mixins.dict import DataClassDictMixin

def optimizer_tagger(cls):
    """Tag serializable optimizers by their class name in lower-case format."""
    return cls.__name__.lower()

class OptimizerTransformIdentifier(enum.StrEnum):
    ADAM = enum.auto()
    SGD = enum.auto()
    PARTITION = enum.auto()

@dataclass(kw_only=True)
class OptimizerTransform(DataClassDictMixin):
    
    def make(self, model: nnx.Module) -> optax.GradientTransformationExtraArgs:
        raise NotImplementedError
    
    def __pre_serialize__(self):
        return self
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Re-build serialization dictionary by creating a `type` field as the first key.
        d = {
            'type': optimizer_tagger(type(self)), # Discriminator field.
            **d,
        }
        return d

@dataclass(kw_only=True)
class Adam(OptimizerTransform):
    """Config for `optax.adam`."""
    learning_rate: float
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    mu_dtype: Any = None
    nesterov: bool = False
    
    def make(self, model: nnx.Module) -> optax.GradientTransformationExtraArgs:
        return optax.adam(
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            eps_root=self.eps_root,
            mu_dtype=self.mu_dtype,
            nesterov=self.nesterov,
        )

@dataclass(kw_only=True)
class SGD(OptimizerTransform):
    """Config for `optax.sgd`."""
    learning_rate: float
    momentum: Optional[float] = None
    nesterov: bool = False
    accumulator_dtype: Optional[Any] = None
    
    def make(self, model: nnx.Module) -> optax.GradientTransformationExtraArgs:
        return optax.sgd(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            accumulator_dtype=self.accumulator_dtype,
        )

type NestedDictStr = dict[str | int, Union[str | int, NestedDictStr]]

@dataclass(kw_only=True)
class Partition(OptimizerTransform):
    # Optimizer transform for every parameter in the model. Keys should be the leaves of the model parameter tree (i.e., the name of each layer or `nnx.Param`).
    transforms: dict[str, 'SupportedOptimizerTransforms'] = field(default_factory=dict)
    pytree: NestedDictStr = field(default_factory=dict)
    
    # See: https://github.com/google-deepmind/optax/issues/1148#issuecomment-2515568364
    def make(self, model: nnx.Module) -> optax.GradientTransformationExtraArgs:
        # {transform_key_name: GradientTransformationExtraArgs}
        transforms = {k: v.make(model) for k,v in self.transforms.items()}
        # Get the model parameters.
        params = nnx.state(model, nnx.Param)
        is_param = lambda x: isinstance(x, nnx.Param)
        # Get the leaves of model parameter tree.
        # name_map_values = [k[0].key for k, _ in jax.tree.flatten_with_path(params, is_leaf=is_param)[0]]
        # name_map_values = [k[-1].key for k, _ in jax.tree.flatten_with_path(params, is_leaf=is_param)[0]]
        name_map_values = [v for _, v in jax.tree.flatten_with_path(self.pytree, is_leaf=is_param)[0]]
        # print(f"{name_map_values = }")
        # Assign transform keys to every leaf in the model parameter tree.
        name_map = jax.tree.unflatten(jax.tree.structure(params, is_leaf=is_param), name_map_values)
        return optax.transforms.partition(
            transforms=transforms,
            param_labels=name_map,
            # param_labels=nnx.State.replace_by_pure_dict(params, self.pytree),
        )

type SupportedOptimizerTransforms = Union[Adam, SGD, Partition]