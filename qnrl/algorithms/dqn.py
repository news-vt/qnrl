"""Deep Q-Learning (DQN) algorithm with experience replay and fixed target.

Experience replay samples a minibatch of environment transitions from a replay buffer.

Fixed target freezes the target Q-network for a set number of steps, after which it resets the target using the current policy model parameters.

Reference:
    V. Mnih et al., “Human-level control through deep reinforcement learning,” Nature, vol. 518, no. 7540, pp. 529–533, Feb. 2015, doi: 10.1038/nature14236.
"""
from dataclasses import dataclass
import os
import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import orbax_utils
import orbax.checkpoint as ocp
from tqdm import tqdm
import optax
import numpy as np
from typing import Any

from .algorithm import Algorithm
from ..checkpoint import CheckpointConfig
from ..modules import SerializableModule
from ..optimizer import OptimizerTransform
from ..buffers import ReplayBuffer
from ..metric import MetricManager


@nnx.jit
def epsilon_schedule_linear(eps_start: float, eps_end: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (eps_end - eps_start) / duration
    return jnp.max(jnp.array([slope * t + eps_start, eps_end]))


def epsilon_greedy_policy(
    env: gym.Env,
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
        q_values = model_policy(observation) # (num_actions,), environment axis is treated as batch index.
        action = q_values.argmax(axis=-1).item()
    return action


@nnx.jit
def update(
    optimizer: nnx.Optimizer,
    model_policy: nnx.Module,
    model_target: nnx.Module,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    ) -> float:
    """Deep Q-learning policy model update."""
    def mse_loss(model, x, y):
        """Generic MSE loss."""
        y_pred = model(x) # (batch_size, num_actions)
        y_pred = y_pred[jnp.arange(y_pred.shape[0]), actions.squeeze()] # (batch_size,)
        return ((y_pred - y) ** 2.).mean()
    
    q_next_target = model_target(next_observations) # (batch_size, n_actions)
    q_next_target = jnp.max(q_next_target, axis=-1) # (batch_size,)
    next_q_value = rewards + (1 - dones) * gamma * q_next_target
    
    loss, grads = nnx.value_and_grad(mse_loss)(model=model_policy, x=observations, y=next_q_value)
    optimizer.update(model_policy, grads) # Update the policy Q-network in-place.
    
    return loss


def train(
    seed: int,
    env: gym.Env,
    model_policy: nnx.Module,
    model_target: nnx.Module,
    optimizer: nnx.Optimizer,
    buffer_size: int,
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
    checkpoint_manager: ocp.CheckpointManager | None = None,
    metric_manager: MetricManager | None = None
    ) -> list[dict[str, Any]]:
    """Deep Q-Learning (DQN) algorithm with experience replay and fixed target.
    
    Experience replay samples a minibatch of environment transitions from a replay buffer.
    
    Fixed target freezes the target Q-network for a set number of steps, after which it resets the target using the current policy model parameters.
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
                # history.append(meta)
                pbar_postfix.update(**meta)
                pbar.set_postfix(pbar_postfix)
                
                # Save metrics.
                if metric_manager is not None:
                    metric_manager.save(t, meta)
            
            # Update models once sufficient environment interactions have been buffered.
            if t > learning_starts:
                
                # Update the policy model via a gradient descent step on a minibatch of samples.
                if t % policy_update_frequency == 0:
                    key, subkey = jax.random.split(key)
                    minibatch = buffer.sample(key=subkey, batch_size=batch_size)
                    loss = update(
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
                    # history.append(meta)
                    pbar_postfix.update(**meta)
                    pbar.set_postfix(pbar_postfix)
                    
                    # Save metrics.
                    if metric_manager is not None:
                        metric_manager.save(t, meta)
                    
                    # Checkpoint model.
                    if checkpoint_manager is not None:
                        state = nnx.state(model_policy) # Get the current model state.
                        save_args = orbax_utils.save_args_from_target(state)
                        checkpoint_manager.save(t, state, save_kwargs={'save_args': save_args}) # Preserve state via checkpoint.
                
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

    return history


# MARK: DQN

@dataclass(kw_only=True)
class DQN(Algorithm):
    """Configuration for `qnrl.algorithms.dqn.train`."""
    # model_policy: SerializableModule # Policy model configuration.
    # model_target: SerializableModule # Target model configuration.
    model: SerializableModule # Target model configuration.
    optimizer: OptimizerTransform
    buffer_size: int = 10_000
    seed: int = 0
    batch_size: int = 128
    total_timesteps: int = 500_000
    learning_starts: int = 10_000
    policy_update_frequency: int = 10 # Apply a policy model update every `t % policy_update_frequency == 0` steps after `t>total_timesteps`.
    target_update_frequency: int = 500 # Apply a target model update every `t % target_update_frequency == 0` steps after `t>total_timesteps`.
    gamma: float = 0.99
    tau: float = 1.0
    eps_start: int = 1
    eps_end: float = 5e-2
    eps_exploration: float = 0.5
    
    def train(self) -> list:

        # Create the environment.
        env = self.env.make()
        
        # # Create models.
        # model_policy = self.model_policy
        # model_target = self.model_target
        # 
        # Create models, target is a copy of policy.
        model_policy = self.model
        graphdef, state = nnx.split(model_policy)
        model_target = nnx.merge(graphdef, state)
        
        # Create optimizers.
        optimizer = nnx.Optimizer(model_policy, self.optimizer.make(model_policy), wrt=nnx.Param)
        
        # Create checkpoint manager.
        if isinstance(self.checkpoint, CheckpointConfig):
            checkpoint_manager = ocp.CheckpointManager(
                directory=os.path.abspath(self.checkpoint.checkpoint_dir),
                checkpointers=ocp.PyTreeCheckpointer(),
                options=self.checkpoint.checkpoint_options.make(),
            )
        else:
            checkpoint_manager = None
        
        # Run the training algorithm.
        history_list = train(
            seed=self.seed,
            env=env,
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
