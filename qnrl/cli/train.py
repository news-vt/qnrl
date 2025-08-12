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

from .. import algorithms
from .. import tools
from ..models.q_learning import QNetworkLinear, DistQNetworkLinear


SupportedDQNModels: TypeAlias = QNetworkLinear
SupportedC51Models: TypeAlias = DistQNetworkLinear

class OptimizerIdentifier(enum.StrEnum):
    adam = "adam"
    sgd = "sgd"

@dataclass(kw_only=True)
class Optimizer:
    optimizer_id: OptimizerIdentifier
    
    def make(self) -> optax.GradientTransformationExtraArgs:
        raise NotImplementedError

@dataclass(kw_only=True)
class Adam:
    """Config for `optax.adam`."""
    optimizer_id: tyro.conf.Fixed[OptimizerIdentifier] = OptimizerIdentifier.adam
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
class SGD:
    """Config for `optax.sgd`."""
    optimizer_id: tyro.conf.Fixed[OptimizerIdentifier] = OptimizerIdentifier.sgd
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


class AlgorithmIdentifier(enum.StrEnum):
    """Identifiers for each algorithm."""
    DQN = "DQN"
    C51 = "C51"


@dataclass(kw_only=True)
class TrainConfig(DataClassDictMixin):
    """Configuration for one of the `qnrl.algorithm.*.train` methods.
    
    This class contains a `run()` method, that is intended to invoke the `qnrl.algorithm.*.train()` function, along with any other training procedures. The `run()` method will be called with no arguments.
    """
    gym: GymConfig # Gymnasium environment configuration.
    algorithm: AlgorithmIdentifier
    
    def run(self):
        raise NotImplementedError


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
            save_train_results(filepath=self.metrics_filepath, d=history_dict)



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



@dataclass(kw_only=True)
class C51(TrainConfig):
    """Configuration for `qnrl.algorithms.c51.train`."""
    algorithm: tyro.conf.Fixed[AlgorithmIdentifier] = AlgorithmIdentifier.C51 # Identifier for algorithm.
    model_policy: DistQNetworkLinear # Policy model configuration.
    model_target: DistQNetworkLinear # Target model configuration.
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
            save_train_results(filepath=self.metrics_filepath, d=history_dict)


@dataclass(kw_only=True)
class C51CartPole(C51):
    """C51 configuration for CartPole-v1."""
    model_policy: DistQNetworkLinear # Policy model configuration.
    model_target: DistQNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="CartPole-v1")) # Gymnasium environment configuration.
    

@dataclass(kw_only=True)
class C51MountainCar(C51):
    """C51 configuration for MountainCar."""
    model_policy: DistQNetworkLinear # Policy model configuration.
    model_target: DistQNetworkLinear # Target model configuration.
    gym: GymConfig = field(default_factory=lambda: GymConfig(env_id="MountainCar-v0")) # Gymnasium environment configuration.

SupportedC51Configs: TypeAlias = Union[C51CartPole, C51MountainCar]




# @dataclass(kw_only=True)
# class EvalConfig(DataClassDictMixin):
#     pass


# @dataclass(kw_only=True)
# class Config(DataClassDictMixin):
#     mode: Annotated[Union[
#         Annotated[TrainConfig, tyro.conf.subcommand("train", prefix_name=False)],
#         Annotated[EvalConfig, tyro.conf.subcommand("eval", prefix_name=False)],
#     ], Discriminator(include_subtypes=True)]


def save_train_results(filepath: str | pathlib.Path, d: dict):
    """Save training result dictionary to JSON file."""
    # Convert filepath to string and format with arguments.
    filepath = str(filepath).format(
        datetime=datetime.datetime.now().isoformat(),
    )
    with open(filepath, 'w+') as f:
        json.dump(d, f, cls=tools.JAXJSONEncoder)


def load_train_results(filepath: str | pathlib.Path) -> dict:
    """Load training result dictionary from JSON file."""
    with open(str(filepath), 'r') as f:
        return json.load(f)


def save_model(path: str | pathlib.Path, model: nnx.Module):
    """Save Flax NNX model."""
    state = nnx.state(model)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"{path}/state", state, save_args=save_args)

def restore_model(path: str | pathlib.Path, model: nnx.Module) -> nnx.Module:
    """Restore a given Flax NNX model to the state designed at the given checkpoint path."""
    state = nnx.state(model) # Get the state of the abstract model.
    # Load state from the saved model.
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(f"{path}/state", item=state)
    # Update the abstract model with the loaded state.
    nnx.update(model, state)
    return model

def load_model(path: str | pathlib.Path, model_create_func: Callable[[], nnx.Module]) -> nnx.Module:
    """Load a Flax NNX model from a given checkpoint path."""
    model = nnx.eval_shape(model_create_func) # Create the model with abstract shapes.
    return restore_model(path, model) # Restore from checkpoint path state.


# def train(config: TrainConfig):
#     print(f"{config = }")
    
#     # Create the environment.
#     env = config.gym.make()
    
#     # Create models.
#     model_policy = config.model_policy.make()
#     model_target = config.model_target.make()
    
#     # Create optimizers.
#     optimizer = nnx.Optimizer(model_policy, config.optimizer.make(), wrt=nnx.Param)
    
#     # Create checkpoint manager.
#     if isinstance(config.checkpoint, CheckpointConfig):
#         checkpoint_manager = ocp.CheckpointManager(
#             directory=os.path.abspath(config.checkpoint.checkpoint_dir),
#             checkpointers=ocp.PyTreeCheckpointer(),
#             options=config.checkpoint.checkpoint_options.make(),
#         )
#     else:
#         checkpoint_manager = None
    
#     # print(f"{checkpoint_manager = }")
#     # return
    
#     # Run the training algorithm.
#     history_list = algorithms.dqn.train(
#         seed=config.seed,
#         env=env,
#         model_policy=model_policy,
#         model_target=model_target,
#         optimizer=optimizer,
#         buffer_size=config.buffer_size,
#         batch_size=config.batch_size,
#         total_timesteps=config.total_timesteps,
#         learning_starts=config.learning_starts,
#         policy_update_frequency=config.policy_update_frequency,
#         target_update_frequency=config.target_update_frequency,
#         gamma=config.gamma,
#         tau=config.tau,
#         eps_start=config.eps_start,
#         eps_end=config.eps_end,
#         eps_exploration=config.eps_exploration,
#         checkpoint_manager=checkpoint_manager,
#     )
    
#     # Convert to dictionary of lists.
#     history_dict = tools.listofdict_to_dictoflist(history_list)
#     # Save training results to file if indicated.
#     if config.metrics_filepath is not None:
#         save_train_results(filepath=config.metrics_filepath, d=history_dict)

class DumpFormat(enum.StrEnum):
    JSON = "JSON"
    YAML = "YAML"


@dataclass(kw_only=True)
class Args(DataClassDictMixin):
    # train_config: DQNCartPole
    train_config: Union[
        SupportedDQNConfigs,
        SupportedC51Configs,
        # DQNCartPole,
        # C51CartPole,
    ]
    dump_config: DumpFormat | None = None


if __name__ == "__main__":
    
    
    # Load config from YAML file.
    default_config = None
    argv = sys.argv[1:]
    if argv and argv[0].endswith((".yaml", ".yml",)):
        with open(argv[0]) as f:
            yaml_config = yaml.safe_load(f) or {}
        default_config = Args.from_dict(yaml_config)
        argv = argv[1:]
    
    # Instantiate the CLI with defaults loaded from YAML file.
    # config = tyro.cli(TrainConfig, config=(tyro.conf.OmitSubcommandPrefixes,), args=argv, default=default_config)
    # config = tyro.cli(Union[
    #     TrainConfig,
    #     DQN,
    #     DQNCartPole,
    # ], config=(tyro.conf.OmitSubcommandPrefixes,), args=argv, default=default_config)
    
    
    args = tyro.cli(Args, config=(tyro.conf.OmitSubcommandPrefixes,), args=argv, default=default_config)
    
    
    if args.dump_config is not None:
        print(f"{args.dump_config = }")
        if args.dump_config == DumpFormat.YAML:
            data = args.to_dict()
            yaml_string = yaml.dump(data, sort_keys=False, default_flow_style=False, indent=2)
            print(yaml_string)
        elif args.dump_config == DumpFormat.JSON:
            data = args.to_dict()
            json_string = json.dumps(data, indent=2)
            print(json_string)
    else:
        # Run training algorithm.
        args.train_config.run()
        # train(config=config)
    
    ###############
    
    # default = tyro.extras.overridable_config_cli(yaml_config)

    # tyro.extras.from_yaml
    
    # default = dataclass_from_dict(Config, yaml_config)
    # default = Config.from_dict(yaml_config)
    
    # default = Config(mode=TrainConfig(
    #     gym=GymConfig(env_id="CartPole-v1"),
    #     model_policy=QNetworkLinearConfig(din=4, num_actions=2, rngs=42, douts=[120, 84]),
    #     model_target=QNetworkLinearConfig(din=4, num_actions=2, rngs=43, douts=[120, 84]),
    #     batch_size=128,
    #     optimizer=Adam(learning_rate=2.5e-4),
    # ))
    # default = TrainConfig(
    #     gym=GymConfig(env_id="CartPole-v1"),
    #     model_policy=QNetworkLinearConfig(din=4, num_actions=2, rngs=42, douts=[120, 84]),
    #     model_target=QNetworkLinearConfig(din=4, num_actions=2, rngs=43, douts=[120, 84]),
    #     batch_size=128,
    #     optimizer=Adam(learning_rate=2.5e-4),
    #     checkpoint=CheckpointConfig(
    #         checkpoint_dir="./checkpoints",
    #         checkpoint_options=CheckpointOptions(
    #             save_interval_steps=1,
    #             max_to_keep=2,
    #         ),
    #     )
    # )
    # default = TrainConfig.from_dict(yaml_config)
    # print(f"{default = }")
    # print()
    # print(f"{default.to_dict() = }")
    # print()
    # print(f"{TrainConfig.from_dict(default.to_dict())}")
    
    # default = load_dataclass_from_dict(Config, yaml_config) if yaml_config is not None else None
    # config = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    # config = tyro.cli(Config, config=(tyro.conf.OmitSubcommandPrefixes,), args=argv, default=default)
    # config = tyro.cli(TrainConfig | EvalConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    # config = tyro.cli(TrainConfig | EvalConfig)
    
    # print(f"{default = }")
    # print(f"{config = }")
    
    # if isinstance(config.mode, TrainConfig):
    #     train(config=config.mode)
    # elif isinstance(config.mode, EvalConfig):
    #     evaluate(config=config.mode)
    # else:
    #     raise ValueError(f"{config.mode} is an unsupported configuration.")