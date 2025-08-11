"""Categorical DQN (C51).

Distributional reinforcement learning variant of DQN.

Reference:
    Bellemare, Marc G., et al. “A Distributional Perspective on Reinforcement Learning.” Proceedings of the 34th International Conference on Machine Learning, edited by Doina Precup and Yee Whye Teh, vol. 70, PMLR, 2017, pp. 449–58, https://doi.org/10.48550/arXiv.1707.06887.
"""
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import optax
import numpy as np
from typing import Any

# from .algorithm import TrainingAlgorithm
from ..buffers import ReplayBuffer


@nnx.jit
def epsilon_schedule_linear(eps_start: float, eps_end: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (eps_end - eps_start) / duration
    return jnp.max(jnp.array([slope * t, eps_end]))


def epsilon_greedy_policy(
    env: gym.Env,
    atoms: jax.Array,
    model_policy: nnx.Module,
    observation: jax.Array,
    t: int, # Current time step.
    duration: int, # Total number of time steps for epsilon decay.
    eps_start: float,
    eps_end: float,
    key: jax.Array,
    ) -> Any:
    """Epsilon-greedy action selection.
    """
    # Decay epsilon based on current time step.
    eps = epsilon_schedule_linear(eps_start=eps_start, eps_end=eps_end, duration=duration, t=t)
    # Action selection.
    if jax.random.uniform(key=key) < eps: # Explore via random action selection.
        action = env.action_space.sample()
    else:
        pmfs = model_policy(observation)
        q_vals = jnp.sum(pmfs * atoms, axis=-1)
        action = jnp.argmax(q_vals, axis=-1).item()
    return action


@nnx.jit
def update(
    atoms: jax.Array,
    num_atoms: int,
    z_min: float,
    z_max: float,
    optimizer: nnx.Optimizer,
    model_policy: nnx.Module,
    model_target: nnx.Module,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    ):
    batch_idx = jnp.arange(actions.shape[0])
    # Generate batch of return-atom distributions for every action.
    next_pmfs = model_target(next_observations) # (batch_size, num_actions, n_atoms)
    # Compute estimated Q-value, which is the expected value of the return-atom distribution.
    next_vals = (next_pmfs * atoms).sum(axis=-1) # (batch_size, num_actions)
    # Determine the optimum action according to max (greedy) Q-value estimate.
    next_action = jnp.argmax(next_vals, axis=-1) # (batch_size,)
    # Get the return-atom distribution corresponding to the optimum action.
    next_pmfs = next_pmfs[batch_idx, next_action] # (batch_size, n_atoms)
    # next_pmfs = next_pmfs[:, next_action] # (batch_size, n_atoms)
    # Compute the atoms of the target distribution and clip to range [z_min, z_max], which are not aligned with the original distribution.
    tz = rewards[:, jnp.newaxis] + gamma * atoms * (1 - dones)[:, jnp.newaxis] # (batch_size, num_atoms)
    tz = jnp.clip(tz, a_min=z_min, a_max=z_max) # Clip to range [z_min, z_max].
    # Project the current atoms to the target atoms.
    dz = atoms[1] - atoms[0] # Delta z.
    b = (tz - z_min) / dz
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Lower (left) bin index.
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Upper (right) bin index.
    dml = (u - b) * next_pmfs # (batch_size, num_atoms), Split the weight onto the lower bin.
    dmu = (b - l) * next_pmfs # (batch_size, num_atoms), Split the weight onto the upper bin.
    
    def project_to_bins(target_pmfs: jax.Array, l: jax.Array, u: jax.Array, dml: jax.Array, dmu: jax.Array):
        """Project an instance of return atoms. Apply to batches using `jax.vmap`."""
        target_pmfs = target_pmfs.at[l.astype(jnp.int32)].add(dml)
        target_pmfs = target_pmfs.at[u.astype(jnp.int32)].add(dmu)
        return target_pmfs
    
    target_pmfs = jax.vmap(project_to_bins)(jnp.zeros_like(next_pmfs), l, u, dml, dmu)
    
    # Categorical cross-entropy loss.
    def cross_entropy_loss(model, x, actions, y):
        batch_idx = jnp.arange(x.shape[0])
        y_hat = model(x)
        # Get the return-atom distribution corresponding to the given actions.
        y_hat = y_hat[batch_idx, actions.squeeze()] # (batch_size, n_atoms)
        y_hat = jnp.clip(y_hat, a_min=1e-5, a_max=1 - 1e-5) # Clip the PMFs for passing to the log, to prevent -inf and 0.
        loss = -jnp.sum(y * jnp.log(y_hat), axis=-1) # (batch_size,)
        return loss.mean()
    
    loss, grads = nnx.value_and_grad(cross_entropy_loss)(model=model_policy, x=observations, actions=actions, y=target_pmfs)
    optimizer.update(model_policy, grads) # Update the policy Q-network in-place.
    
    return loss


def train(
    env: gym.Env,
    buffer_size: int,
    seed: int,
    atoms: jax.Array,
    num_atoms: int,
    z_min: float,
    z_max: float,
    model_policy: nnx.Module,
    model_target: nnx.Module,
    optimizer: nnx.Optimizer,
    batch_size: int,
    total_timesteps: int, # Max number of training steps.
    learning_starts: int, # Training step index where model training commences.
    policy_update_frequency: int, # Apply a policy model update every `t % policy_update_frequency == 0` steps after `t>total_timesteps`.
    target_update_frequency: int, # Apply a target model update every `t % target_update_frequency == 0` steps after `t>total_timesteps`.
    gamma: float,
    tau: float,
    eps_start: float,
    eps_end: float,
    eps_exploration: float,
    ) -> list[dict[str, Any]]:
    """Categorical DQN (C51).

    Distributional reinforcement learning variant of DQN.
    """
    # Training history.
    history: list[dict[str, Any]] = []
    
    # Create a buffer for environment interactions.
    buffer = ReplayBuffer(buffer_size=buffer_size, env=env)
    
    # Apply several additional wrappers on the environment to ensure functionality.
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track statistics for every episode.
    env = gym.wrappers.Autoreset(env) # Automatically reset the environment (continuous training).
    
    # Seed random generators.
    np.random.seed(seed)
    env.action_space.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    # Reset the environment.
    observation, _ = env.reset(seed=seed)
    
    # Iteratively step through environment.
    pbar = tqdm(range(total_timesteps), unit='step', leave=True)
    pbar_postfix = dict() # Current progress bar postfix content.
    
    try:
        for t in pbar:
            
            # Action selection.
            key, subkey = jax.random.split(key)
            action = epsilon_greedy_policy(
                env=env,
                atoms=atoms,
                model_policy=model_policy,
                observation=observation,
                t=t,
                duration=int(eps_exploration * total_timesteps),
                eps_start=eps_start,
                eps_end=eps_end,
                key=subkey,
                )
            
            # Interaction within environment using selected actions.
            next_observation, reward, done, truncation, info = env.step(action)
            
            # Push interaction onto buffer.
            buffer.add(observation, next_observation, action, reward, done)
            
            # Override current observation to be the next observation for subsequent loop iteration.
            observation = next_observation.copy()
            
            # Record episode statistics.
            if "episode" in info:
                meta = dict(
                    episode_step=t,
                    episode_reward=info['episode']['r'],
                    episode_length=info['episode']['l'],
                    episode_time=info['episode']['t'],
                )
                history.append(meta)
                pbar_postfix.update(**meta)
                pbar.set_postfix(pbar_postfix)
            
            # Update models once sufficient environment interactions have been buffered.
            if t > learning_starts:
                
                # Update the policy model via a gradient descent step on a minibatch of samples.
                if t % policy_update_frequency == 0:
                    key, subkey = jax.random.split(key)
                    minibatch = buffer.sample(key=subkey, batch_size=batch_size)
                    loss = update(
                        atoms,
                        num_atoms,
                        z_min,
                        z_max,
                        optimizer,
                        model_policy,
                        model_target,
                        minibatch.observations,
                        minibatch.actions,
                        minibatch.next_observations,
                        minibatch.rewards,
                        minibatch.dones,
                        gamma,
                    )
                    
                    # Update the history and progress bar.
                    meta = dict(
                        loss_step=t,
                        loss=loss,
                    )
                    history.append(meta)
                    pbar_postfix.update(**meta)
                    pbar.set_postfix(pbar_postfix)
                
                # Update the target model by replacing its parameters with those of the policy model.
                if t % target_update_frequency == 0:
                    params_policy = nnx.state(model_policy, nnx.Param) # Get the learnable parameters of the policy model.
                    params_target = nnx.state(model_target, nnx.Param) # Get the learnable parameters of the target model.
                    params_update = optax.incremental_update(params_policy, params_target, tau) # Apply soft update rule for target network weights.
                    nnx.update(model_target, params_update)
    
    # Gracefully handle `Ctrl+C`.
    except KeyboardInterrupt:
        pass

    return history
