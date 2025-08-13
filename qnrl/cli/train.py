import sys
import os
from dataclasses import dataclass, field
import pathlib
from typing import Any, Callable, Optional, Union, Annotated, TypeAlias, Type, ClassVar, Literal
import datetime
import json
import yaml
import json
import enum

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import orbax_utils
import optax
import gymnasium as gym
import tyro
import orbax.checkpoint as ocp
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.types import Discriminator
import rich
import rich.pretty

from .. import algorithms
from .. import tools
from ..models.q_learning import QNetworkLinear, DistQNetworkLinear

# MARK: Optimizers

class OptimizerIdentifier(enum.StrEnum):
    ADAM = enum.auto()
    SGD =enum.auto()

@dataclass(kw_only=True)
class Optimizer:
    optimizer_id: OptimizerIdentifier
    
    def make(self) -> optax.GradientTransformationExtraArgs:
        raise NotImplementedError

@dataclass(kw_only=True)
class Adam(Optimizer):
    """Config for `optax.adam`."""
    optimizer_id: tyro.conf.Fixed[OptimizerIdentifier] = OptimizerIdentifier.ADAM
    learning_rate: float
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    mu_dtype: Any = None
    nesterov: bool = False
    
    def make(self) -> optax.GradientTransformationExtraArgs:
        return optax.adam(
            learning_rate=self.learning_rate,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            eps_root=self.eps_root,
            mu_dtype=self.mu_dtype,
            nesterov=self.nesterov,
        )

@dataclass(kw_only=True)
class SGD(Optimizer):
    """Config for `optax.sgd`."""
    optimizer_id: tyro.conf.Fixed[OptimizerIdentifier] = OptimizerIdentifier.SGD
    learning_rate: float
    momentum: Optional[float] = None
    nesterov: bool = False
    accumulator_dtype: Optional[Any] = None
    
    def make(self) -> optax.GradientTransformationExtraArgs:
        return optax.sgd(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            accumulator_dtype=self.accumulator_dtype,
        )

SupportedOptimizers: TypeAlias = Union[Adam, SGD]


# MARK: Environments
@dataclass(kw_only=True)
class GymConfig:
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

# MARK: Checkpoints

@dataclass(kw_only=True)
class CheckpointOptions:
    """Configuration for `orbax.checkpoint.CheckpointManagerOptions`."""
    save_interval_steps: int = 1
    max_to_keep: int | None = None
    enable_async_checkpointing: bool = True
    
    def make(self) -> ocp.CheckpointManagerOptions:
        return ocp.CheckpointManagerOptions(
            save_interval_steps=self.save_interval_steps,
            max_to_keep=self.max_to_keep,
            enable_async_checkpointing=self.enable_async_checkpointing,
        )

@dataclass(kw_only=True)
class CheckpointConfig:
    """Checkpoint configuration."""
    checkpoint_dir: str | pathlib.Path
    checkpoint_options: CheckpointOptions


@dataclass(kw_only=True)
class NoneConfig:
    """Empty configuration, mirrors behavior of `None`.
    
    Use like:
    >>> if isinstance(cfg, NoneConfig): print("config is None")
    """
    pass


# MARK: Training config

class AlgorithmIdentifier(enum.StrEnum):
    """Identifiers for each algorithm."""
    DQN = enum.auto()
    C51 = enum.auto()


# Lists of supported model classes for each algorithm.
SupportedDQNModels: TypeAlias = QNetworkLinear
SupportedC51Models: TypeAlias = DistQNetworkLinear


@dataclass(kw_only=True)
class TrainConfig(DataClassDictMixin):
    """Configuration for one of the `qnrl.algorithm.*.train` methods.
    
    This class contains a `run()` method, that is intended to invoke the `qnrl.algorithm.*.train()` function, along with any other training procedures. The `run()` method will be called with no arguments.
    """
    gym: GymConfig # Gymnasium environment configuration.
    algorithm: AlgorithmIdentifier
    
    def run(self):
        raise NotImplementedError

# MARK: DQN

@dataclass(kw_only=True)
class DQN(TrainConfig):
    """Configuration for `qnrl.algorithms.dqn.train`."""
    algorithm: tyro.conf.Fixed[AlgorithmIdentifier] = AlgorithmIdentifier.DQN # Identifier for algorithm.
    model_policy: SupportedDQNModels # Policy model configuration.
    model_target: SupportedDQNModels # Target model configuration.
    optimizer: SupportedOptimizers
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
    
    checkpoint: Union[
        Annotated[CheckpointConfig, tyro.conf.subcommand("checkpoint-on")],
        Annotated[NoneConfig, tyro.conf.subcommand("checkpoint-off")],
    ] = field(default_factory=NoneConfig) # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.
    
    def run(self):
        # Validate the metrics file path (if any).
        if self.metrics_filepath is not None:
            os.makedirs(self.metrics_filepath.parent, exist_ok=True)
        
        # Create the environment.
        env = self.gym.make()
        
        # Create models.
        model_policy = self.model_policy
        model_target = self.model_target
        
        # Create optimizers.
        optimizer = nnx.Optimizer(model_policy, self.optimizer.make(), wrt=nnx.Param)
        
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



@dataclass(kw_only=True)
class DQNCartPole(DQN):
    """DQN configuration for CartPole-v1."""
    model_policy: QNetworkLinear # Policy model configuration.
    model_target: QNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="CartPole-v1")) # Gymnasium environment configuration.

@dataclass(kw_only=True)
class DQNMountainCar(DQN):
    """DQN configuration for MountainCar-v0."""
    model_policy: QNetworkLinear # Policy model configuration.
    model_target: QNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="MountainCar-v0")) # Gymnasium environment configuration.

SupportedDQNConfigs: TypeAlias = Union[DQNCartPole, DQNMountainCar]


# MARK: C51

@dataclass(kw_only=True)
class C51(TrainConfig):
    """Configuration for `qnrl.algorithms.c51.train`."""
    algorithm: tyro.conf.Fixed[AlgorithmIdentifier] = AlgorithmIdentifier.C51 # Identifier for algorithm.
    model_policy: SupportedC51Models # Policy model configuration.
    model_target: SupportedC51Models # Target model configuration.
    optimizer: SupportedOptimizers
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
    
    checkpoint: Union[
        Annotated[CheckpointConfig, tyro.conf.subcommand("checkpoint-on")],
        Annotated[NoneConfig, tyro.conf.subcommand("checkpoint-off")],
    ] = field(default_factory=NoneConfig) # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.
    
    def run(self):
        # Validate the metrics file path (if any).
        if self.metrics_filepath is not None:
            os.makedirs(self.metrics_filepath.parent, exist_ok=True)
        
        # Create the environment.
        env = self.gym.make()
        
        # Create models.
        model_policy = self.model_policy
        model_target = self.model_target
        
        # Create optimizers.
        optimizer = nnx.Optimizer(model_policy, self.optimizer.make(), wrt=nnx.Param)
        
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


@dataclass(kw_only=True)
class C51CartPole(C51):
    """C51 configuration for CartPole-v1."""
    model_policy: DistQNetworkLinear # Policy model configuration.
    model_target: DistQNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="CartPole-v1")) # Gymnasium environment configuration.


@dataclass(kw_only=True)
class C51MountainCar(C51):
    """C51 configuration for MountainCar-v0."""
    model_policy: DistQNetworkLinear # Policy model configuration.
    model_target: DistQNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="MountainCar-v0")) # Gymnasium environment configuration.

SupportedC51Configs: TypeAlias = Union[C51CartPole, C51MountainCar]


class DumpFormat(enum.StrEnum):
    JSON = enum.auto()
    YAML = enum.auto()
    PRETTY = enum.auto()


# MARK: CLI

@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    """Training CLI."""
    train_config: Union[
        SupportedDQNConfigs,
        SupportedC51Configs,
    ]
    dump_config: DumpFormat | None = None


if __name__ == "__main__":
    
    # Load config from YAML file.
    default_config = None
    argv = sys.argv[1:]
    if argv:
        # Load from YAML.
        if argv[0].endswith((".yaml", ".yml",)):
            with open(argv[0]) as f:
                config = yaml.safe_load(f) or {}
        # Load from JSON.
        elif argv[0].endswith((".json",)):
            with open(argv[0]) as f:
                config = json.load(f) or {}
        
        # Load the configuration object.
        default_config = Args.from_dict(config)
        argv = argv[1:]
    
    # Instantiate the CLI with defaults loaded from YAML file.
    args = tyro.cli(Args, config=(tyro.conf.OmitSubcommandPrefixes, tyro.conf.EnumChoicesFromValues,), args=argv, default=default_config)
    
    # Dump configuration to various formats.
    if args.dump_config is not None:
        match args.dump_config:
            case DumpFormat.YAML:
                data = args.to_dict()
                yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
                print(yaml_string)
            case DumpFormat.JSON:
                data = args.to_dict()
                rich.print_json(data=data, indent=2)
            case DumpFormat.PRETTY:
                data = args.to_dict()
                rich.pretty.pprint(data, expand_all=True)
    else:
        # Run training algorithm.
        args.train_config.run()