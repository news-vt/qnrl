from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import gymnasium as gym
from mashumaro.mixins.dict import DataClassDictMixin
import dill


# MARK: Environments
@dataclass(kw_only=True)
class EnvConfig(DataClassDictMixin):
    """Config for `gymnasium.make`."""
    # Environment string name ID.
    env_id: str
    # Environment render mode.
    render_mode: str | None = None
    # Maximum number of steps per episode.
    max_episode_steps: int | None = None
    # Environment-specific keyword arguments.
    extra_args: dict[str, int | float | str] | None = None
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        kwargs_default = dict(
            id=self.env_id,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps,
            **(self.extra_args if self.extra_args is not None else {}),
        )
        
        return gym.make(
            **{**kwargs_default, **kwargs}, # Apply default keyword arguments with per-make overrides.
        )


@dataclass(kw_only=True)
class EnvWrapperConfig(DataClassDictMixin):
    """Config for `gymnasium.wrappers.*`."""
    env: 'EnvConfig | EnvWrapperConfig'
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = self.env.make(**kwargs)
        return env

@dataclass(kw_only=True)
class ScaleObservation(EnvWrapperConfig):
    """Transforms observation via `obs/scale`."""
    scale: float | list[float]
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / jnp.array(self.scale), env.observation_space)
        return env

@dataclass(kw_only=True)
class ObservationExpandDims(EnvWrapperConfig):
    """Transforms observation by inserting extra dimension along the given axes using `jax.numpy.expand_dims`."""
    axis: int | tuple[int,...]
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.TransformObservation(env, lambda obs: jnp.expand_dims(jnp.array(obs), axis=self.axis), env.observation_space)
        return env

@dataclass(kw_only=True)
class ObservationDiscreteOneHot(EnvWrapperConfig):
    """Transforms a discrete observation into a one-hot encoded representation.
    
    If a desired number of classes are not provided then it will be inferred from the wrapped environment observation space.
    """
    num_classes: int | None = None
    
    def make(self, **kwargs) -> gym.Env:
        env = super().make(**kwargs)
        assert isinstance(env.observation_space, gym.spaces.Discrete), "Only environment with discrete observation spaces are supported for one-hot transformation."
        if self.num_classes is None:
            self.num_classes = int(env.observation_space.n)
        env = gym.wrappers.TransformObservation(
            env=env,
            func=lambda obs: jax.nn.one_hot(obs, num_classes=self.num_classes, dtype=int),
            observation_space=gym.spaces.MultiBinary(n=self.num_classes),
            )
        return env