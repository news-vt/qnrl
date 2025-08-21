from dataclasses import dataclass
import os
from typing import Any

from flax import nnx
from flax.training import orbax_utils
import gymnasium as gym
import optax
import orbax.checkpoint as ocp
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

from .algorithm import Algorithm
from ..checkpoint import CheckpointConfig
from ..modules import QuantumNativeQNetworkSplit, SerializableModule
from ..optimizer import OptimizerTransform
from ..buffers import ReplayBuffer
from ..metric import MetricManager

@nnx.jit
def epsilon_schedule_linear(eps_start: float, eps_end: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (eps_end - eps_start) / duration
    return jnp.max(jnp.array([slope * t, eps_end]))


# TODO refactor for QnRL
def epsilon_greedy_policy(
    env: gym.Env,
    model_policy: QuantumNativeQNetworkSplit,
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
        # pmfs = model_policy(observation) # (batch,n_actions,)
        # action = jnp.argmax(pmfs, axis=-1).item() # ()
        pmfs = model_policy.call_gen_action_dist(observation) # (batch,n_actions,)
        action = jnp.argmax(pmfs, axis=-1).item() # ()
    return action


# TODO QnRL loss should compare the quantum states of the return-atom dist between the policy model and the target model w.r.t. the policy model's action distribution. Input the policy model's predicted action distribution, then prepare 2 return-atom circuits using the policy and target model's circuit_weights. Then run quantum wasserstein distance or quantum KL divergence to compare the quantum states of the two return-atom registers. 

@nnx.jit
def update(
    optimizer: nnx.Optimizer,
    model_policy: QuantumNativeQNetworkSplit,
    model_target: QuantumNativeQNetworkSplit,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    ):
    
    atoms = model_policy.atoms
    num_atoms = model_policy.n_returns
    z_min = model_policy.z_min
    z_max = model_policy.z_max
    
    # batch_idx = jnp.arange(actions.shape[0])
    
    # Generate batch of return-atom distributions for every action.
    # next_pmfs = model_target(next_observations) # (batch_size, num_actions, n_atoms)
    
    # # Compute estimated Q-value, which is the expected value of the return-atom distribution.
    # next_vals = (next_pmfs * atoms).sum(axis=-1) # (batch_size, num_actions)
    
    # # Determine the optimum action according to max (greedy) Q-value estimate.
    # next_action = jnp.argmax(next_vals, axis=-1) # (batch_size,)
    
    # # Get the return-atom distribution corresponding to the optimum action.
    # next_pmfs = next_pmfs[batch_idx, next_action] # (batch_size, n_atoms)
    
    
    ######
    # # Generate return PMF for next observation, which is weighted based on expectation for each action.
    # # i.e., the `next_pmfs` is a weighted sum across all the individual action return-atom PMFs.
    # next_pmfs = model_target.call_gen_return_dist(next_observations) # (batch_size,n_atoms)
    next_action_pmfs = model_target.call_gen_action_dist(next_observations) # (batch,n_actions,)
    next_actions = jnp.argmax(next_action_pmfs, axis=-1).item() # (batch,)
    next_actions_onehot = nnx.one_hot(next_actions, num_classes=model_policy.n_actions)
    next_pmfs = model_target.call_gen_return_dist_NEW(next_observations, next_actions_onehot) # (batch_size,n_atoms)
    ######


    # next_pmfs = next_pmfs[:, next_action] # (batch_size, n_atoms)
    # Compute the atoms of the target distribution and clip to range [z_min, z_max], which are not aligned with the original distribution.
    tz = rewards[:, jnp.newaxis] + gamma * atoms * (1 - dones)[:, jnp.newaxis] # (batch_size, num_atoms)
    tz = jnp.clip(tz, a_min=z_min, a_max=z_max) # Clip to range [z_min, z_max].
    # Project the current atoms to the target atoms.
    dz = atoms[1] - atoms[0] # Delta z.
    b = (tz - z_min) / dz
    l = jnp.clip(jnp.floor(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Lower (left) bin index.
    u = jnp.clip(jnp.ceil(b), a_min=0, a_max=num_atoms - 1) # (batch_size, num_atoms), Upper (right) bin index.
    # Handle the case where `b_j` is exactly an integer via `(l == u).astype(u.dtype)`.
    dml = ((u + (l == u).astype(u.dtype)) - b) * next_pmfs # (batch_size, num_atoms), Split the weight onto the lower bin.
    dmu = (b - l) * next_pmfs # (batch_size, num_atoms), Split the weight onto the upper bin.
    
    def project_to_bins(target_pmfs: jax.Array, l: jax.Array, u: jax.Array, dml: jax.Array, dmu: jax.Array):
        """Project an instance of return atoms. Apply to batches using `jax.vmap`."""
        target_pmfs = target_pmfs.at[l.astype(jnp.int32)].add(dml)
        target_pmfs = target_pmfs.at[u.astype(jnp.int32)].add(dmu)
        return target_pmfs
    
    target_pmfs = jax.vmap(project_to_bins)(jnp.zeros_like(next_pmfs), l, u, dml, dmu)
    
    # Categorical cross-entropy loss.
    def cross_entropy_loss(model, x, probs_actions, y):
        # batch_idx = jnp.arange(x.shape[0])
        # y_hat = model(x)
        # # Get the return-atom distribution corresponding to the given actions.
        # y_hat = y_hat[batch_idx, actions.squeeze()] # (batch_size, n_atoms)
        # 
        # y_hat = model.call_gen_return_dist(x) # (batch_size, n_atoms)
        y_hat = model.call_gen_return_dist_NEW(x, probs_actions) # (batch_size, n_atoms)
        # 
        y_hat = jnp.clip(y_hat, a_min=1e-5, a_max=1 - 1e-5) # Clip the PMFs for passing to the log, to prevent -inf and 0.
        loss = -jnp.sum(y * jnp.log(y_hat), axis=-1) # (batch_size,)
        return loss.mean()
    
    # Get action PMF outside of gradient computation.
    probs_actions = model_policy.call_gen_action_dist(observations)
    
    loss, grads = nnx.value_and_grad(cross_entropy_loss)(model=model_policy, x=observations, probs_actions=probs_actions, y=target_pmfs)
    optimizer.update(model_policy, grads) # Update the policy Q-network in-place.
    
    return loss

# @nnx.jit
# def update(
#     optimizer: nnx.Optimizer,
#     model_policy: nnx.Module,
#     model_target: nnx.Module,
#     observations: jax.Array,
#     # actions: jax.Array,
#     next_observations: jax.Array,
#     # rewards: jax.Array,
#     # dones: jax.Array,
#     # gamma: float,
#     ) -> float:
#     """Deep Q-learning policy model update."""
#     def loss_fn(model, x, y):
#         """KL-divergence loss."""
#         # y_hat = model(x) # (batch, n_actions), Probability distribution over the actions.
#         # y_hat = jnp.log(y_hat) # Convert to log-prob.
#         # loss = optax.kl_divergence(log_predictions=y_hat, targets=y) # (batch,)
#         # return loss.mean()
#         ####
#         y_hat = model(x) # (batch, n_actions), Probability distribution over the actions.
#         y_hat = jnp.clip(y_hat, a_min=1e-5, a_max=1 - 1e-5) # Clip the PMFs for passing to the log, to prevent -inf and 0.
#         loss = -jnp.sum(y * jnp.log(y_hat), axis=-1) # (batch_size,)
#         return loss.mean()
    
#     target_next_pmfs = model_target(next_observations) # (batch_size, n_actions)
    
#     loss, grads = nnx.value_and_grad(loss_fn)(model=model_policy, x=observations, y=target_next_pmfs)
#     optimizer.update(model_policy, grads) # Update the policy Q-network in-place.
    
#     return loss


def train(
    seed: int,
    env: gym.Env,
    model_policy: QuantumNativeQNetworkSplit,
    model_target: QuantumNativeQNetworkSplit,
    optimizer: nnx.Optimizer,
    # 
    buffer_size: int,
    batch_size: int,
    total_timesteps: int, # Max number of training steps.
    learning_starts: int, # Training step index where model training commences.
    policy_update_frequency: int, # Apply a policy model update every `t % policy_update_frequency == 0` steps after `t>total_timesteps`.
    target_update_frequency: int, # Apply a target model update every `t % target_update_frequency == 0` steps after `t>total_timesteps`.
    gamma: float,
    tau: float,
    # 
    eps_start: float,
    eps_end: float,
    eps_exploration: float,
    checkpoint_manager: ocp.CheckpointManager | None = None,
    metric_manager: MetricManager | None = None
    ):
    """Quantum-native reinforcement learning (QnRL).
    
    A variant of distributional reinforcement learning (DRL) algorithm C51 leveraging quantum superposition as the distributional generator.
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
                history.append(meta)
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
                        # atoms,
                        # num_atoms,
                        # z_min,
                        # z_max,
                        optimizer=optimizer,
                        model_policy=model_policy,
                        model_target=model_target,
                        observations=minibatch.observations,
                        actions=minibatch.actions,
                        next_observations=minibatch.next_observations,
                        rewards=minibatch.rewards,
                        dones=minibatch.dones,
                        gamma=gamma,
                    )
                    
                    # Update the history and progress bar.
                    meta = dict(
                        loss_step=t,
                        loss=loss,
                    )
                    history.append(meta)
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


def eval(
    seed: int,
    env: gym.Env,
    model_policy: QuantumNativeQNetworkSplit,
    total_timesteps: int, # Max number of training steps.
    ) -> list:

    # Apply several additional wrappers on the environment to ensure functionality.
    env = gym.wrappers.RecordEpisodeStatistics(env) # Track statistics for every episode.
    
    history = []

    observation, _ = env.reset() # Reset the environment.
    
    # Iteratively step through environment.
    pbar = tqdm(range(total_timesteps), unit='step', leave=True)
    pbar_postfix = dict() # Current progress bar postfix content.
    for t in pbar:
        pmfs = model_policy.call_gen_action_dist(observation) # (batch,n_actions,)
        action = jnp.argmax(pmfs, axis=-1).item() # ()
        next_observation, reward, done, truncation, info = env.step(action)
        
        meta = {
            'pmfs': pmfs,
            'action': action,
            'reward': reward,
            'observation': observation,
            'next_observation': next_observation,
            'done': done,
        }
        history.append(meta)
        pbar_postfix.update({
            'reward': reward,
        })
        pbar.set_postfix(pbar_postfix)
        
        # Record episode statistics.
        if "episode" in info:
            meta = dict(
                episode_step=t,
                episode_reward=info['episode']['r'],
                episode_length=info['episode']['l'],
                episode_time=info['episode']['t'],
            )
            history.append(meta)
            pbar_postfix.update({
                'reward': reward,
            })
            pbar.set_postfix(pbar_postfix)
        
        # Update current observation with next observation.
        observation = next_observation.copy()
        
        if done:
            break
    
    return history


@dataclass(kw_only=True)
class QnRL(Algorithm):
    """Configuration for `qnrl.algorithms.c51.train`."""
    model_policy: QuantumNativeQNetworkSplit # Policy model configuration.
    model_target: QuantumNativeQNetworkSplit # Target model configuration.
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
        
        # Create models.
        model_policy = self.model_policy
        model_target = self.model_target
        
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
    
    def eval(self) -> list:
        
        # Create the environment.
        env = self.env.make()
        
        # Create models.
        model_policy = self.model_policy

        # Create checkpoint manager and restore latest model state.
        if isinstance(self.checkpoint, CheckpointConfig):
            checkpoint_manager = ocp.CheckpointManager(
                directory=os.path.abspath(self.checkpoint.checkpoint_dir),
                checkpointers=ocp.PyTreeCheckpointer(),
                options=self.checkpoint.checkpoint_options.make(),
            )
            
            # Restore model state from checkpoint.
            original_state = nnx.state(model_policy)
            step_to_restore = checkpoint_manager.latest_step()
            restored_state = checkpoint_manager.restore(step_to_restore, items=original_state)
            nnx.update(model_policy, restored_state)
        else:
            checkpoint_manager = None
        
        history_list = eval(
            seed=self.seed,
            env=env,
            model_policy=model_policy,
            total_timesteps=self.total_timesteps,
        )
        
        return history_list