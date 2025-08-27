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
            save_interval_steps=self.save_interval_steps,
            max_to_keep=self.max_to_keep,
            enable_async_checkpointing=self.enable_async_checkpointing,
        )

@dataclass(kw_only=True)
class CheckpointConfig(DataClassDictMixin):
    """Checkpoint configuration."""
    checkpoint_dir: str | pathlib.Path
    checkpoint_options: CheckpointOptions
    
    def make(self) -> ocp.CheckpointManager:
        """Build an `orbax.checkpoint.CheckpointManager` instance."""
        checkpoint_manager = ocp.CheckpointManager(
            directory=os.path.abspath(self.checkpoint_dir),
            checkpointers=ocp.PyTreeCheckpointer(),
            options=self.checkpoint_options.make(),
        )
        return checkpoint_manager


@dataclass(kw_only=True)
class NoneConfig:
    """Empty configuration, mirrors behavior of `None`.
    
    Use like:
    >>> if isinstance(cfg, NoneConfig): print("config is None")
    """
    pass