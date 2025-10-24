from dataclasses import dataclass
import pathlib
import os

import orbax.checkpoint as ocp
from mashumaro.mixins.dict import DataClassDictMixin

# MARK: Checkpoints

@dataclass(kw_only=True)
class CheckpointOptions(DataClassDictMixin):
    """Configuration for `orbax.checkpoint.CheckpointManagerOptions`."""
    save_interval_steps: int = 1
    max_to_keep: int | None = None
    enable_async_checkpointing: bool = True
    
    def make(self) -> ocp.CheckpointManagerOptions:
        """Build an `orbax.checkpoint.CheckpointManagerOptions` instance."""
        return ocp.CheckpointManagerOptions(
            save_decision_policy=ocp.checkpoint_managers.save_decision_policy.FixedIntervalPolicy(self.save_interval_steps),
            preservation_policy=ocp.checkpoint_managers.preservation_policy.LatestN(self.max_to_keep),
            enable_async_checkpointing=self.enable_async_checkpointing,
        )

@dataclass(kw_only=True)
class CheckpointConfig(DataClassDictMixin):
    """Checkpoint configuration."""
    checkpoint_dir: str | pathlib.Path
    checkpoint_options: CheckpointOptions | None = None
    
    def make(self) -> ocp.CheckpointManager:
        """Build an `orbax.checkpoint.CheckpointManager` instance."""
        
        # Create a registry for state and metrics metadata.
        # We freeze these options here to conform the training code to a specific pattern.
        handler_registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        handler_registry.add('state', ocp.args.StandardSave, ocp.StandardCheckpointHandler)
        handler_registry.add('state', ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        handler_registry.add('metrics', ocp.args.JsonSave, ocp.JsonCheckpointHandler)
        handler_registry.add('metrics', ocp.args.JsonRestore, ocp.JsonCheckpointHandler)
        
        # Build the checkpoint manager.
        checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.abspath(self.checkpoint_dir),
            options=self.checkpoint_options.make() if self.checkpoint_options is not None else None,
            handler_registry=handler_registry,
        )
        return checkpoint_manager


@dataclass(kw_only=True)
class NoneConfig:
    """Empty configuration, mirrors behavior of `None`.
    
    Use like:
    >>> if isinstance(cfg, NoneConfig): print("config is None")
    """
    pass