from dataclasses import dataclass

import jax.numpy as jnp
import gymnasium as gym
from mashumaro.mixins.dict import DataClassDictMixin


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
    
    def make(self) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        kwargs = (self.extra_args if self.extra_args is not None else {})
        return gym.make(
            id=self.env_id,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps,
            **kwargs,
        )


@dataclass(kw_only=True)
class EnvWrapperConfig(DataClassDictMixin):
    """Config for `gymnasium.wrappers.*`."""
    env: 'EnvConfig | EnvWrapperConfig'
    
    def make(self) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = self.env.make()
        return env


@dataclass(kw_only=True)
class ScaleObservation(EnvWrapperConfig):
    """Transforms observation via `obs/scale`."""
    scale: float | list[float]
    
    def make(self) -> gym.Env:
        """Wrapper for `gymnasium.make` using config arguments."""
        env = super().make()
        env = gym.wrappers.TransformObservation(env, lambda obs: obs / jnp.array(self.scale), env.observation_space)
        return env