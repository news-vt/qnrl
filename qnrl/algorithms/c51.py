"""Categorical DQN (C51).

Distributional reinforcement learning variant of DQN.

Reference:
    M. G. Bellemare, W. Dabney, and R. Munos, “A distributional perspective on reinforcement learning,” in Proceedings of the 34th international conference on machine learning, D. Precup and Y. W. Teh, Eds., in Proceedings of machine learning research, vol. 70. PMLR, Aug. 2017, pp. 449–458. doi: 10.48550/arXiv.1707.06887.
"""
from dataclasses import dataclass
import os
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import orbax_utils
from tqdm import tqdm
import optax
import numpy as np
from typing import Any, Callable
import orbax.checkpoint as ocp

from .algorithm import Algorithm
from ..checkpoint import CheckpointConfig
from ..modules import SerializableModule
from ..optimizer import OptimizerTransform
from ..buffers import ReplayBuffer
from ..metric import MetricManager
from ..tools import restore_model_from_checkpointmanager

# https://docs.cleanrl.dev/rl-algorithms/c51/#implemented-variants
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_jax.py
# https://github.com/archsyscall/DistRL-TensorFlow2?tab=readme-ov-file#qrdqn
# https://github.com/archsyscall/DistRL-TensorFlow2/blob/master/C51/C51.py


@nnx.jit
def epsilon_schedule_linear(eps_start: float, eps_end: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (eps_end - eps_start) / duration
    return jnp.max(jnp.array([slope * t + eps_start, eps_end]))


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
    
    ## Compute the atoms of the target distribution and clip to range [z_min, z_max], which are not aligned with the original distribution.
    # 
    # `tz` are the atoms shifted by the discounted reward.
    tz = rewards[:, jnp.newaxis] + gamma * atoms * (1 - dones)[:, jnp.newaxis] # (batch_size, num_atoms)
    tz = jnp.clip(tz, a_min=z_min, a_max=z_max) # Clip to range [z_min, z_max].
    
    ## Project the current atoms to the target atoms.
    #
    # `dz` is the width between atoms (assumed to be constant across all atoms).
    dz = atoms[1] - atoms[0] # Delta z.
    # `b` represents how far `tz` has been shifted from the original atoms.
    # The value of `b` is a floating-point number that falls between two atoms.
    # For example, if the shifted atom 0 is `b[0] = 0.31` then it is between the original atom 0 and atom 1, therefore `l[0] = 0`, `u[0] = 1`, `dml_scalar[0] = 0.69`, `dmu_scalar[0] = 0.31`. So we multiply the original PMF probability at each atom index by `dml_scalar` for the fraction that should be reassigned to the lower bin, and multiply by `dmu_scalar` for the fraction assigned to the upper bin.
    b = (tz - z_min) / dz
    # `l` are the indices of the atoms to the left of `b`.
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Lower (left) bin index.
    # `u` are the indices of the atoms to the right of `b`.
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Upper (right) bin index.
    # Here we split the original PMF `next_pmfs` into lower / upper bins by multiplying a fraction of the original probability that should be reassigned to each bin.
    dml_scalar = ((u + (l == u).astype(u.dtype)) - b) # Handle the case where `b_j` is exactly an integer via `(l == u).astype(u.dtype)`.
    dml = dml_scalar * next_pmfs # (batch_size, num_atoms), Split the weight onto the lower bin.
    dmu_scalar = (b - l)
    dmu = dmu_scalar * next_pmfs # (batch_size, num_atoms), Split the weight onto the upper bin.
    
    def project_to_bins(target_pmfs: jax.Array, l: jax.Array, u: jax.Array, dml: jax.Array, dmu: jax.Array):
        """Project an instance of return atoms.
        
        This function takes atom bin indices `l` (lower bins) and `u` (upper bins) and shift amounts for each bin `dml` (lower bin shift amount) and `dmu` (upper bin shift amount).
        
        Apply to batches using `jax.vmap`.
        """
        target_pmfs = target_pmfs.at[l.astype(jnp.int32)].add(dml)
        target_pmfs = target_pmfs.at[u.astype(jnp.int32)].add(dmu)
        return target_pmfs
    
    target_pmfs = jax.vmap(project_to_bins)(jnp.zeros_like(next_pmfs), l, u, dml, dmu)
    
    # Compute other metrics prior to update.
    def compute_other_metrics(model, x, actions, y) -> dict:
        batch_idx = jnp.arange(x.shape[0])
        y_hat = model(x)
        # Get the return-atom distribution corresponding to the given actions.
        y_hat = y_hat[batch_idx, actions.squeeze()] # (batch_size, n_atoms)
        y = jnp.clip(y, a_min=1e-5, a_max=1 - 1e-5) # Clip the PMFs for passing to the log, to prevent -inf and 0.
        #
        # Cross entropy.
        cross_entropy = -y * jnp.log(y_hat) # (batch_size, n_atoms)
        cross_entropy = cross_entropy.sum(axis=-1).mean()
        
        # KL Divergence.
        kl_div = jax.vmap(jax.scipy.special.kl_div)(p=y, q=y_hat) # (batch_size, n_atoms,)
        kl_div = kl_div.sum(axis=-1).mean()
        
        # Add all additional metrics to a dictionary.
        metrics = dict(
            cross_entropy=cross_entropy,
            kl_div=kl_div,
        )
        return metrics
    
    # Compute other metrics
    metrics = compute_other_metrics(model=model_policy, x=observations, actions=actions, y=target_pmfs)
    
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
    
    return loss, metrics


def train(
    seed: int,
    env: gym.Env,
    atoms: jax.Array,
    num_atoms: int,
    z_min: float,
    z_max: float,
    model_policy: nnx.Module,
    model_target: nnx.Module,
    optimizer: nnx.Optimizer,
    buffer_size: int,
    batch_size: int,
    total_timesteps: int, # Max number of training steps.
    learning_starts: int, # Training step index where model training commences.
    policy_update_frequency: int, # Apply a policy model update every `t % policy_update_frequency == 0` steps after `t>learning_starts`.
    target_update_frequency: int, # Apply a target model update every `t % target_update_frequency == 0` steps after `t>learning_starts`.
    gamma: float,
    tau: float,
    eps_start: float,
    eps_end: float,
    eps_exploration: float,
    checkpoint_manager: ocp.CheckpointManager | None = None,
    metric_manager: MetricManager | None = None
    ) -> list[dict[str, Any]]:
    """Categorical DQN (C51).

    Distributional reinforcement learning variant of DQN.
    """
    # Training history.
    history: list[dict[str, Any]] = []
    
    # Track latest episode and loss metadata bundles.
    episode_meta = dict()
    loss_meta = dict()
    
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
                episode_meta = dict(
                    episode_step=t,
                    episode_reward=info['episode']['r'],
                    episode_length=info['episode']['l'],
                    episode_time=info['episode']['t'],
                )
                history.append(episode_meta)
                pbar_postfix.update(**episode_meta)
                pbar.set_postfix(pbar_postfix)
                
                # Save metrics.
                if metric_manager is not None:
                    metric_manager.save(t, episode_meta)
            
            # Update models once sufficient environment interactions have been buffered.
            if t > learning_starts:
                
                # Update the policy model via a gradient descent step on a minibatch of samples.
                if t % policy_update_frequency == 0:
                    key, subkey = jax.random.split(key)
                    minibatch = buffer.sample(key=subkey, batch_size=batch_size)
                    loss, metrics = update(
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
                    
                    # Update the history and progress bar with loss information.
                    loss_meta = dict(
                        loss_step=t,
                        loss=loss,
                    )
                    history.append(loss_meta)
                    pbar_postfix.update(**loss_meta)
                    pbar.set_postfix(pbar_postfix)
                    
                    # Update the history and progress bar with metric information.
                    # This is separate from loss so that it can be correlated with a different time step variable.
                    metric_meta = dict(
                        metric_step=t,
                        **metrics,
                    )
                    history.append(metric_meta)
                    pbar_postfix.update(**metric_meta)
                    pbar.set_postfix(pbar_postfix)
                    
                    # Save metrics.
                    if metric_manager is not None:
                        metric_manager.save(t, {**loss_meta, **metric_meta})
                    
                    # Checkpoint model.
                    if checkpoint_manager is not None:
                        state = nnx.state(model_policy) # Get the current model state.
                        checkpoint_manager.save(
                            step=t,
                            args=ocp.args.Composite(
                                state=ocp.args.StandardSave(state), 
                                metrics=ocp.args.JsonSave(dict(
                                    loss=float(loss_meta['loss'].item()) if 'loss' in loss_meta else None,
                                    episode_reward=float(episode_meta['episode_reward']) if 'episode_reward' in episode_meta else None,
                                    **{k: float(v) for k,v in metrics.items()},
                                )),
                            ),
                        ) # Preserve state via checkpoint.
                
                # Update the target model by replacing its parameters with those of the policy model.
                if t % target_update_frequency == 0:
                    params_policy = nnx.state(model_policy, nnx.Param) # Get the learnable parameters of the policy model.
                    params_target = nnx.state(model_target, nnx.Param) # Get the learnable parameters of the target model.
                    params_update = optax.incremental_update(params_policy, params_target, tau) # Apply soft update rule for target network weights.
                    nnx.update(model_target, params_update)
    
    # Gracefully handle `Ctrl+C`.
    except KeyboardInterrupt:
        pass
    
    # Wait until checkpoint async operations have completed.
    if metric_manager is not None:
        metric_manager.wait_until_finished()
    
    # Wait until checkpoint async operations have completed.
    if checkpoint_manager is not None:
        checkpoint_manager.wait_until_finished()
        checkpoint_manager.close()

    return history


def eval(
    seed: int,
    env: gym.Env,
    model_policy: nnx.Module,
    atoms: jax.Array,
    total_timesteps: int, # Max number of training steps.
    env_wrappers: list[Callable[[gym.Env], gym.Env]] = [],
    ) -> list:

    # Apply several additional wrappers on the environment to ensure functionality.
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track statistics for every episode.
    for wrapper_func in env_wrappers:
        env = wrapper_func(env)
    
    # Seed random generators.
    np.random.seed(seed)
    env.action_space.seed(seed)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    history = []

    observation, _ = env.reset() # Reset the environment.
    
    # Iteratively step through environment.
    pbar = tqdm(range(total_timesteps), unit='step', leave=True)
    pbar_postfix = dict() # Current progress bar postfix content.
    for t in pbar:
        pmfs = model_policy(observation)
        q_vals = jnp.sum(pmfs * atoms, axis=-1)
        action = jnp.argmax(q_vals, axis=-1).item()
        # 
        next_observation, reward, done, truncated, info = env.step(action)
        
        meta = {
            'pmfs': pmfs,
            'action': action,
            'reward': reward,
            'observation': observation,
            'next_observation': next_observation,
            'done': done,
        }
        history.append(meta)
        
        # Record episode statistics.
        if "episode" in info:
            meta = dict(
                episode_step=t,
                episode_reward=info['episode']['r'],
                episode_length=info['episode']['l'],
                episode_time=info['episode']['t'],
            )
            history.append(meta)
            pbar_postfix.update(meta)
            pbar.set_postfix(pbar_postfix)
        
        # Update current observation with next observation.
        observation = next_observation.copy()
        
        if done or truncated:
            break
    
    # Close environment.
    # This is necessary if there are wrappers that operate once the environment is closed.
    env.close()
    
    return history



# MARK: C51

@dataclass(kw_only=True)
class C51(Algorithm):
    """Configuration for `qnrl.algorithms.c51.train`."""
    model: SerializableModule # Policy model configuration.
    optimizer: OptimizerTransform
    num_atoms: int
    z_min: float
    z_max: float
    buffer_size: int = 10_000
    batch_size: int = 128
    total_timesteps: int = 500_000
    learning_starts: int = 10_000
    policy_update_frequency: int = 10 # Apply a policy model update every `t % policy_update_frequency == 0` steps after `t>learning_starts`.
    target_update_frequency: int = 500 # Apply a target model update every `t % target_update_frequency == 0` steps after `t>learning_starts`.
    gamma: float = 0.99
    tau: float = 1.0
    eps_start: int = 1
    eps_end: float = 5e-2
    eps_exploration: float = 0.5
    
    def train(self) -> list:
        
        # Create the environment.
        env = self.env.make()
        
        # Create models, target is a copy of policy.
        model_policy = self.model
        graphdef, state = nnx.split(model_policy)
        model_target = nnx.merge(graphdef, state)
        
        # Create optimizers.
        optimizer = nnx.Optimizer(model_policy, self.optimizer.make(model_policy), wrt=nnx.Param)
        
        # Create list of return atoms.
        atoms = jnp.asarray(np.linspace(start=self.z_min, stop=self.z_max, num=self.num_atoms))
        
        # Create checkpoint manager.
        if isinstance(self.checkpoint, CheckpointConfig):
            checkpoint_manager = self.checkpoint.make()
        else:
            checkpoint_manager = None
        
        # Run the training algorithm.
        history_list = train(
            seed=self.seed,
            env=env,
            atoms=atoms,
            num_atoms=self.num_atoms,
            z_min=self.z_min,
            z_max=self.z_max,
            model_policy=model_policy,
            model_target=model_target,
            optimizer=optimizer,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            total_timesteps=self.total_timesteps,
            learning_starts=self.learning_starts,
            policy_update_frequency=self.policy_update_frequency,
            target_update_frequency=self.target_update_frequency,
            gamma=self.gamma,
            tau=self.tau,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_exploration=self.eps_exploration,
            checkpoint_manager=checkpoint_manager,
            metric_manager=self.metric_manager,
        )

        return history_list

    def eval(self, **kwargs) -> list:
        
        # Create the environment.
        env = self.env.make()
        
        # Create models.
        model_policy = self.model
        
        # Create checkpoint manager and restore latest model state (if one exists).
        if self.checkpoint is not None:
            checkpoint_manager = self.checkpoint.make()
            model_policy = restore_model_from_checkpointmanager(model=model_policy, checkpoint_manager=checkpoint_manager)
        
        # Create list of return atoms.
        atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.num_atoms)
        
        # Run evaluation.
        kwargs_default = dict(
            seed=self.seed,
            env=env,
            model_policy=model_policy,
            atoms=atoms,
            total_timesteps=self.total_timesteps,
        )
        history_list = eval(**{**kwargs_default, **kwargs})
        
        return history_list