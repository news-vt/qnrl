import sys
import os
from dataclasses import dataclass, field
import pathlib
from typing import Any, Callable, Optional, Union, Annotated, TypeAlias, Type
import datetime
import json
import yaml

import jax
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
from ..models.q_learning import QNetworkLinear


@dataclass
class RngsConfig:
    """Config for `jax.nnx.Rngs`."""
    default: int | list[int]
    rngs: dict[str, int | list[int] | dict[str, int | list[int]]] | None = None

    def make(self) -> nnx.Rngs:
        rngs = (self.rngs if self.rngs is not None else {})
        return nnx.Rngs(default=self.default, **rngs)

@dataclass
class QNetworkLinearConfig:
    """Config for `qnrl.models.q_learning.QNetworkLinear`."""
    din: int
    num_actions: int
    # rngs: RngsConfig
    rngs: int
    douts: list[int] = field(default_factory=list)
    
    def make(self) -> 'QNetworkLinear':
        return QNetworkLinear(
            din=self.din,
            num_actions=self.num_actions,
            # rngs=self.rngs.make(),
            rngs=nnx.Rngs(self.rngs),
            douts=self.douts,
        )


@dataclass
class AdamOptimizerConfig:
    """Config for `optax.adam`."""
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

@dataclass
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


@dataclass
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

@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    checkpoint_dir: str | pathlib.Path
    checkpoint_options: CheckpointOptions


@dataclass
class NoneConfig:
    """Empty configuration, mirrors behavior of `None`.
    
    Use like:
    >>> if isinstance(cfg, NoneConfig): print("config is None")
    """
    pass


@dataclass
class TrainConfig(DataClassDictMixin):
    """Configuration for `qnrl.algorithms.dqn.train`."""
    gym: GymConfig # Gymnasium environment configuration.
    model_policy: QNetworkLinearConfig # Policy model configuration.
    model_target: QNetworkLinearConfig # Target model configuration.
    optimizer: AdamOptimizerConfig # Adam optimizer configuration.
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
    
    checkpoint: Union[
        Annotated[CheckpointConfig, tyro.conf.subcommand("checkpoint-on")],
        Annotated[NoneConfig, tyro.conf.subcommand("checkpoint-off")],
    ] = field(default_factory=NoneConfig) # Checkpoint configuration. If this option is not specified, then checkpointing will be disabled.

    metrics_filepath: pathlib.Path | None = None # Path for metric file with optional format flags. If this option is not specified, then the metrics will not be saved.


# @dataclass
# class EvalConfig(DataClassDictMixin):
#     pass


# @dataclass
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


def train(config: TrainConfig):
    print(f"{config = }")
    
    # Create the environment.
    env = config.gym.make()
    
    # Create models.
    model_policy = config.model_policy.make()
    model_target = config.model_target.make()
    
    # Create optimizers.
    optimizer = nnx.Optimizer(model_policy, config.optimizer.make(), wrt=nnx.Param)
    
    # Create checkpoint manager.
    if isinstance(config.checkpoint, CheckpointConfig):
        checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.abspath(config.checkpoint.checkpoint_dir),
            checkpointers=ocp.PyTreeCheckpointer(),
            options=config.checkpoint.checkpoint_options.make(),
        )
    else:
        checkpoint_manager = None
    
    # print(f"{checkpoint_manager = }")
    # return
    
    # Run the training algorithm.
    history_list = algorithms.dqn.train(
        seed=config.seed,
        env=env,
        model_policy=model_policy,
        model_target=model_target,
        optimizer=optimizer,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        total_timesteps=config.total_timesteps,
        learning_starts=config.learning_starts,
        policy_update_frequency=config.policy_update_frequency,
        target_update_frequency=config.target_update_frequency,
        gamma=config.gamma,
        tau=config.tau,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        eps_exploration=config.eps_exploration,
        checkpoint_manager=checkpoint_manager,
    )
    
    # Convert to dictionary of lists.
    history_dict = tools.listofdict_to_dictoflist(history_list)
    # Save training results to file if indicated.
    if config.metrics_filepath is not None:
        save_train_results(filepath=config.metrics_filepath, d=history_dict)



if __name__ == "__main__":
    
    
    # Load config from YAML file.
    default_config = None
    argv = sys.argv[1:]
    if argv and argv[0].endswith((".yaml", ".yml",)):
        with open(argv[0]) as f:
            yaml_config = yaml.safe_load(f) or {}
        default_config = TrainConfig.from_dict(yaml_config)
        argv = argv[1:]
    
    # Instantiate the CLI with defaults loaded from YAML file.
    config = tyro.cli(TrainConfig, config=(tyro.conf.OmitSubcommandPrefixes,), args=argv, default=default_config)
    
    print(f"{config = }")
    
    # Run training algorithm.
    train(config=config)
    
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
    #     optimizer=AdamOptimizerConfig(learning_rate=2.5e-4),
    # ))
    # default = TrainConfig(
    #     gym=GymConfig(env_id="CartPole-v1"),
    #     model_policy=QNetworkLinearConfig(din=4, num_actions=2, rngs=42, douts=[120, 84]),
    #     model_target=QNetworkLinearConfig(din=4, num_actions=2, rngs=43, douts=[120, 84]),
    #     batch_size=128,
    #     optimizer=AdamOptimizerConfig(learning_rate=2.5e-4),
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