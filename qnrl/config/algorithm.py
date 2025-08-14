from dataclasses import dataclass, field
import enum
import pathlib
import os
from typing import Optional, Union, Any, TypeAlias

import jax
from flax import nnx
import optax
from mashumaro.mixins.dict import DataClassDictMixin

from .checkpoint import *
from .optimizer import *
from .gym import *
from ..modules import *
from .. import tools
from .. import algorithms


# MARK: Training config

class AlgorithmIdentifier(enum.StrEnum):
    """Identifiers for each algorithm."""
    DQN = enum.auto()
    C51 = enum.auto()


@dataclass(kw_only=True)
class AlgorithmConfig(DataClassDictMixin):
    """Configuration for one of the `qnrl.algorithm.*.train` methods.
    
    This class contains a `run()` method, that is intended to invoke the `qnrl.algorithm.*.train()` function, along with any other training procedures. The `run()` method will be called with no arguments.
    """
    name: str  = field(init=False) # Unique name for configuration.
    gym: GymConfig # Gymnasium environment configuration.
    algorithm: AlgorithmIdentifier = field(init=False)
    
    def run(self) -> dict:
        raise NotImplementedError
    
    class Config(BaseConfig):
        discriminator = Discriminator(  # <- add discriminator
            field="name",
            include_subtypes=True,
        )

# MARK: DQN

@dataclass(kw_only=True)
class DQN(AlgorithmConfig):
    """Configuration for `qnrl.algorithms.dqn.train`."""
    name: str = field(default='dqn', init=False)
    algorithm: AlgorithmIdentifier = field(default=AlgorithmIdentifier.DQN, init=False) # Identifier for algorithm.
    model_policy: SerializableModule # Policy model configuration.
    model_target: SerializableModule # Target model configuration.
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
    
    metrics_filepath: pathlib.Path | None = None # Path for metric file with optional format flags. If this option is not specified, then the metrics will not be saved.
    
    checkpoint: CheckpointConfig | None = None  # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.
    
    def run(self) -> dict:
        # Validate the metrics file path (if any).
        if self.metrics_filepath is not None:
            os.makedirs(self.metrics_filepath.parent, exist_ok=True)
        
        # Create the environment.
        env = self.gym.make()
        
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
        history_list = algorithms.dqn.train(
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
        )
        
        # Convert to dictionary of lists.
        history_dict = tools.listofdict_to_dictoflist(history_list)
        # Save training results to file if indicated.
        if self.metrics_filepath is not None:
            tools.save_train_results(filepath=self.metrics_filepath, d=history_dict)
        
        return history_dict



@dataclass(kw_only=True)
class DQNCartPole(DQN):
    """DQN configuration for CartPole-v1."""
    name: str = field(default='dqn-cartpole-v1', init=False)
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="CartPole-v1")) # Gymnasium environment configuration.

@dataclass(kw_only=True)
class DQNMountainCar(DQN):
    """DQN configuration for MountainCar-v0."""
    name: str = field(default='dqn-mountaincar-v0', init=False)
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="MountainCar-v0")) # Gymnasium environment configuration.

SupportedDQNConfigs: TypeAlias = Union[DQNCartPole, DQNMountainCar]


# MARK: C51

@dataclass(kw_only=True)
class C51(AlgorithmConfig):
    """Configuration for `qnrl.algorithms.c51.train`."""
    name: str = field(default='c51', init=False)
    algorithm: AlgorithmIdentifier = field(default=AlgorithmIdentifier.C51, init=False) # Identifier for algorithm.
    model_policy: SerializableModule # Policy model configuration.
    model_target: SerializableModule # Target model configuration.
    optimizer: OptimizerTransform
    num_atoms: int
    z_min: float
    z_max: float
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
    
    metrics_filepath: pathlib.Path | None = None # Path for metric file with optional format flags. If this option is not specified, then the metrics will not be saved.
    
    checkpoint: CheckpointConfig | None = None # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.
    
    def run(self) -> dict:
        # Validate the metrics file path (if any).
        if self.metrics_filepath is not None:
            os.makedirs(self.metrics_filepath.parent, exist_ok=True)
        
        # Create the environment.
        env = self.gym.make()
        
        # Create models.
        model_policy = self.model_policy
        model_target = self.model_target
        
        # Create optimizers.
        optimizer = nnx.Optimizer(model_policy, self.optimizer.make(model_policy), wrt=nnx.Param)
        
        # Create list of return atoms.
        atoms = jnp.linspace(start=self.z_min, stop=self.z_max, num=self.num_atoms)
        
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
        history_list = algorithms.c51.train(
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
        )
        
        # Convert to dictionary of lists.
        history_dict = tools.listofdict_to_dictoflist(history_list)
        # Save training results to file if indicated.
        if self.metrics_filepath is not None:
            tools.save_train_results(filepath=self.metrics_filepath, d=history_dict)

        return history_dict


@dataclass(kw_only=True)
class C51CartPole(C51):
    """C51 configuration for CartPole-v1."""
    name: str = field(default='c51-cartpole-v1', init=False)
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="CartPole-v1")) # Gymnasium environment configuration.


@dataclass(kw_only=True)
class C51MountainCar(C51):
    """C51 configuration for MountainCar-v0."""
    name: str = field(default='c51-mountaincar-v1', init=False)
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="MountainCar-v0")) # Gymnasium environment configuration.

SupportedC51Configs: TypeAlias = Union[C51CartPole, C51MountainCar]