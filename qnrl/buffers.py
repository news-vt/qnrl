from dataclasses import dataclass
import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def get_observation_space_shape(observation_space: gym.spaces.Space):
    if isinstance(observation_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return observation_space.shape
    elif isinstance(observation_space, gym.spaces.Box):
        return observation_space.shape
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_action_space_shape(action_space: gym.spaces.Space):
    if isinstance(action_space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return (int(len(action_space.nvec)),)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return action_space.shape
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_action_pmf_space_shape(action_space: gym.spaces.Space):
    if isinstance(action_space, gym.spaces.Discrete):
        return (1,action_space.n,)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return (int(len(action_space.nvec)),action_space.n,)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


@dataclass
class ReplayBufferSlice:
    """Collection of elements from a replay buffer."""
    observations: jax.Array
    next_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    truncations: jax.Array


@dataclass # This allows the buffer to be displayed properly using `IPython.display`.
class ReplayBuffer:
    """Circular replay buffer written in JAX.
    
    Supports both `gym.Env` and `gym.vector.VectorEnv` environments, the latter used for sample diversification.
    
    In the case of a `gym.vector.VectorEnv`, the environment axis within the buffer shape `(buffer_size, num_envs, ...)` is also treated as a random sample index, along with the buffer size axis. This means sampling a minibatch will generate random indices from both `buffer_size` and `num_envs` axes, resulting in a sample with shape `(batch_size, ...)`.
    
    Inspired by: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/buffers.py
    """
    observations: jax.Array
    next_observations: jax.Array
    actions: jax.Array
    rewards: jax.Array
    terminations: jax.Array
    truncations: jax.Array
    
    def __init__(self,
        buffer_size: int,
        env: gym.Env | gym.vector.VectorEnv,
        ):
        super().__init__()
        self.idx = 0 # Index within buffer.
        self.is_full = False # Indicates if the buffer has reached full capacity, also indicates when circular replacement is used.
        self.buffer_size = buffer_size
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.observation_shape = get_observation_space_shape(self.observation_space)
        self.action_shape = get_action_space_shape(self.action_space)
        # Create sub-buffers for each environment element.
        if isinstance(env, gym.vector.VectorEnv):
            self.num_envs = env.num_envs
            self.observations = jnp.zeros(shape=(self.buffer_size, *self.observation_shape), dtype=self.observation_space.dtype)
        else:
            self.num_envs = 1
            self.observations = jnp.zeros(shape=(self.buffer_size, self.num_envs, *self.observation_shape), dtype=self.observation_space.dtype)
        self.next_observations = jnp.zeros_like(self.observations)
        self.actions = jnp.zeros(shape=(self.buffer_size, *self.action_shape), dtype=self.action_space.dtype)
        self.rewards = jnp.zeros(shape=(self.buffer_size, self.num_envs), dtype=jnp.float32)
        self.terminations = jnp.zeros(shape=(self.buffer_size, self.num_envs), dtype=bool)
        self.truncations = jnp.zeros(shape=(self.buffer_size, self.num_envs), dtype=bool)
    
    def reset(self):
        """Reinitialize the buffer with all zeroes."""
        self.idx = 0 # Index within buffer.
        self.is_full = False # Indicates if the buffer has reached full capacity, also indicates when circular replacement is used.
        self.observations = jnp.zeros_like(self.observations)
        self.next_observations = jnp.zeros_like(self.next_observations)
        self.actions = jnp.zeros_like(self.actions)
        self.rewards = jnp.zeros_like(self.rewards)
        self.terminations = jnp.zeros_like(self.terminations)
        self.truncations = jnp.zeros_like(self.truncations)
    
    def add(self,
        observation: jax.Array,
        next_observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        ):
        """Insert an environment interaction onto the buffer in circular fashion."""
        
        # Add elements onto buffer.
        self.observations = self.observations.at[self.idx].set(jnp.asarray(observation).reshape((*self.observations.shape[1:],)))
        self.next_observations = self.next_observations.at[self.idx].set(jnp.asarray(next_observation).reshape((*self.observations.shape[1:],)))
        self.actions = self.actions.at[self.idx].set(jnp.asarray(action).reshape((*self.actions.shape[1:],)))
        self.rewards = self.rewards.at[self.idx].set(jnp.asarray(reward).reshape((*self.rewards.shape[1:],)))
        self.terminations = self.terminations.at[self.idx].set(jnp.asarray(terminated).reshape((*self.terminations.shape[1:],)))
        self.truncations = self.truncations.at[self.idx].set(jnp.asarray(truncated).reshape((*self.truncations.shape[1:],)))
        
        self.idx += 1 # Increment buffer index.
        if self.idx >= self.buffer_size:
            self.is_full = True # Indicate that the buffer has reached capacity.
            self.idx = 0 # Reset index if buffer is full (circular buffer).
    
    def sample(self, key: ArrayLike, batch_size: int) -> ReplayBufferSlice:
        """Collect randomized samples from the buffer."""
        # Collect random batch indices.
        key, subkey = jax.random.split(key) # Generate a new random key, use `subkey` for calls to jax.random, preserve `self.key` for future splits.
        if self.is_full:
            idx_batch = (jax.random.randint(subkey, shape=(batch_size,), minval=1, maxval=self.buffer_size) + self.idx) % self.buffer_size
        else:
            idx_batch = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=self.idx)

        # Collect random environment indices.
        key, subkey = jax.random.split(key) # Generate a new random key, use `subkey` for calls to jax.random, preserve `self.key` for future splits.
        idx_env = jax.random.randint(subkey, shape=(len(idx_batch),), minval=0, maxval=self.num_envs)
        
        # Return a sample of the batched elements.
        return self[idx_batch, idx_env]
    
    def __getitem__(self, idx: int | list[int] | tuple[ArrayLike, ...] | ArrayLike) -> ReplayBufferSlice:
        """Access a sample of the buffer via index notation."""
        data = dict(
            observations=self.observations[idx],
            next_observations=self.next_observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            terminations=self.terminations[idx],
            truncations=self.truncations[idx],
        )
        return ReplayBufferSlice(**data)


@dataclass
class PmfReplayBufferSlice(ReplayBufferSlice):
    """Collection of elements from a replay buffer, which includes action probability mass function (PMF)."""
    action_pmfs: jax.Array

@dataclass # This allows the buffer to be displayed properly using `IPython.display`.
class PmfReplayBuffer(ReplayBuffer):
    """Variant of circular replay buffer with support for action probability mass functions (PMFs).
    """
    action_pmfs: jax.Array
    
    def __init__(self,
        buffer_size: int,
        env: gym.Env | gym.vector.VectorEnv,
        ):
        super().__init__(buffer_size=buffer_size, env=env)
        self.action_pmf_shape = get_action_pmf_space_shape(self.action_space)
        self.action_pmfs = jnp.zeros(shape=(self.buffer_size, *self.action_pmf_shape), dtype=jnp.float32)
    
    def reset(self):
        """Reinitialize the buffer with all zeroes."""
        super().reset()
        
        self.action_pmfs = jnp.zeros_like(self.action_pmfs)
    
    def add(self,
        observation: jax.Array,
        next_observation: jax.Array,
        action: jax.Array,
        action_pmf: jax.Array,
        reward: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        ):
        """Insert an environment interaction onto the buffer in circular fashion."""
        
        # Add elements onto buffer.
        self.action_pmfs = self.action_pmfs.at[self.idx].set(jnp.asarray(action_pmf).reshape((*self.action_pmfs.shape[1:],)))
        
        super().add(
            observation=observation,
            next_observation=next_observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
    
    def __getitem__(self, idx: int | list[int] | tuple[ArrayLike, ...] | ArrayLike) -> PmfReplayBufferSlice:
        """Access a sample of the buffer via index notation."""
        data = dict(
            observations=self.observations[idx],
            next_observations=self.next_observations[idx],
            actions=self.actions[idx],
            action_pmfs=self.action_pmfs[idx],
            rewards=self.rewards[idx],
            terminations=self.terminations[idx],
            truncations=self.truncations[idx],
        )
        return PmfReplayBufferSlice(**data)