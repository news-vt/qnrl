from dataclasses import dataclass

import gymnasium as gym
from mashumaro.mixins.dict import DataClassDictMixin

# MARK: Environments
@dataclass(kw_only=True)
class GymConfig(DataClassDictMixin):
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
            max_episode_steps=self.max_episode_steps,
            **kwargs,
        )