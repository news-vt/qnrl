from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import Discriminator


def env_config_tagger(cls):
    """Tag serializable environment configurations by their class name."""
    return cls.__name__


@dataclass(kw_only=True)
class EnvironmentBase(DataClassDictMixin):
    """Base environment configuration class for serialize/deserialize. Do not use this class directly. Instead, use the `EnvConfig` class for creating environments, or `EnvWrapperConfig` for wrapping environment.
    """
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        raise NotImplementedError
    
    class Config(BaseConfig):
        discriminator = Discriminator(
            field="type",
            include_subtypes=True,
            variant_tagger_fn=env_config_tagger,
        )
    
    def __pre_serialize__(self):
        return self
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Re-build serialization dictionary by creating a `type` field as the first key.
        d = {
            'type': env_config_tagger(type(self)), # Discriminator field.
            **d,
        }
        return d

@dataclass(kw_only=True)
class EnvConfig(EnvironmentBase):
    """Config for `gymnasium.make`."""
    # Environment string name ID.
    env_id: str
    # Environment render mode.
    render_mode: str | None = None
    # Maximum number of steps per episode.
    max_episode_steps: int | None = None
    # Environment-specific keyword arguments.
    extra_args: dict[str, int | float | str | tuple | list | dict | None] | None = None
    
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
class EnvWrapperConfig(EnvironmentBase):
    """Config for `gymnasium.wrappers.*`."""
    # env: 'EnvConfig | EnvWrapperConfig'
    env: EnvironmentBase
    
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
        observation_space = gym.spaces.MultiBinary(n=self.num_classes)
        env = gym.wrappers.TransformObservation(
            env=env,
            func=lambda obs: jax.nn.one_hot(obs, num_classes=self.num_classes, dtype=observation_space.dtype),
            observation_space=observation_space,
            )
        return env

@dataclass(kw_only=True)
class ObservationDiscreteUnravel(EnvWrapperConfig):
    """Transforms a discrete 1-dimensional observation into an N-dimension observation using `numpy.unravel_index`.
    
    For example, this transformation is useful in 2-dimensional grid-world environments (e.g., CliffWalking-v1 with shape `(4,12)`) that have observations that report a singular grid index value (e.g., `47`) corresponding to a grid location (e.g., `(3,11)`). This transformation will "unravel" the singular index value into the multi-dimensional grid coordinate location.
    """
    shape: tuple[int,...]
    order: Literal['C', 'F'] = 'C' # 'C' row-major style, 'F' column-major.
    
    def make(self, **kwargs) -> gym.Env:
        env = super().make(**kwargs)
        assert isinstance(env.observation_space, gym.spaces.Discrete), "Only environment with discrete observation spaces are supported for unravel transformation."
        observation_space = gym.spaces.MultiDiscrete(nvec=list(self.shape))
        env = gym.wrappers.TransformObservation(
            env=env,
            func=lambda obs: jnp.asarray(np.unravel_index(obs, self.shape, order=self.order)),
            observation_space=observation_space,
            )
        return env

@dataclass(kw_only=True)
class FrameStackObservation(EnvWrapperConfig):
    """Wrapper for `gymnasium.wrappers.FrameStackObservation`."""
    stack_size: int
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.FrameStackObservation(env=env, stack_size=self.stack_size)
        return env

@dataclass(kw_only=True)
class GrayscaleObservation(EnvWrapperConfig):
    """Wrapper for `gymnasium.wrappers.GrayscaleObservation`."""
    keep_dim: bool = False
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.GrayscaleObservation(env=env, keep_dim=self.keep_dim)
        return env

@dataclass(kw_only=True)
class ResizeObservation(EnvWrapperConfig):
    """Wrapper for `gymnasium.wrappers.ResizeObservation`."""
    shape: tuple[int, int]
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.ResizeObservation(env=env, shape=self.shape)
        return env



@dataclass(kw_only=True)
class AtariPreprocessing(EnvWrapperConfig):
    """Wrapper for `gymnasium.wrappers.AtariPreprocessing`."""
    noop_max: int = 30
    frame_skip: int = 4
    screen_size: int | tuple[int, int] = 84
    terminal_on_life_loss: bool = False
    grayscale_obs: bool = True
    grayscale_newaxis: bool = False
    scale_obs: bool = False
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.AtariPreprocessing(
            env=env,
            noop_max=self.noop_max,
            frame_skip=self.frame_skip,
            screen_size=self.screen_size,
            terminal_on_life_loss=self.terminal_on_life_loss,
            grayscale_obs=self.grayscale_obs,
            grayscale_newaxis=self.grayscale_newaxis,
            scale_obs=self.scale_obs,
        )
        return env


class FireResetEnv(gym.Wrapper):
    """Based on `cleanrl.cleanrl_utils.atari_wrappers.FireResetEnv`.
    
    See: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/atari_wrappers.py#L94
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

@dataclass(kw_only=True)
class FireResetEnvWrapper(EnvWrapperConfig):
    """Config wrapper for `FireResetEnv`."""
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = FireResetEnv(env=env)
        return env


@dataclass(kw_only=True)
class FlattenObservation(EnvWrapperConfig):
    """Config wrapper for `gymnasium.wrappers.FlattenObservation`."""
    
    def make(self, **kwargs) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make(**kwargs)
        env = gym.wrappers.FlattenObservation(env=env)
        return env