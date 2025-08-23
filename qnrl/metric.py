from dataclasses import dataclass, field
import pathlib
import os
import sys
import shutil
from typing import Any, Literal

from mashumaro.mixins.dict import DataClassDictMixin

from . import tools


@dataclass(kw_only=True)
class MetricManager(DataClassDictMixin):
    # path: pathlib.Path
    metric_dir: pathlib.Path # Path to unique directory where to store the files.
    metric_file_type: Literal['json', 'yaml', 'npy', 'npz'] # File format for metrics.
    save_kwargs: dict = field(default_factory=dict)
    save_interval_steps: int = 1 # Save metrics when `step % save_interval_steps == 0`.
    max_to_keep: int | None = 1 # Maximum number of metrics files to keep. If `None` then keep everything.
    
    _history: list[dict[str, Any]] = field(default_factory=list, init=False)
    _existing_metric_dirs: list[pathlib.Path] = field(default_factory=list, init=False)
    _metric_filename: str = field(default='', init=False)
    _latest_step_tracked: int | None = field(default=None, init=False)
    _latest_step_untracked: int | None = field(default=None, init=False)
    
    def get_history(self) -> list[dict[str, Any]]:
        """Return history as list of time-sorted dictionary metric entries."""
        return self._history

    def get_metrics(self) -> dict[str, Any]:
        """Return metrics as dictionary of lists."""
        return tools.listofdict_to_dictoflist(self._history) # list[dict] -> dict[str, Any]
    
    def __post_init__(self):
        
        # Ensure max to keep is valid.
        if self.max_to_keep is not None and self.max_to_keep <= 0:
            raise ValueError(f"Invalid max to keep {self.max_to_keep}")
        
        # Ensure save interval steps is valid.
        if self.save_interval_steps <= 0:
            raise ValueError(f"Invalid save interval steps {self.save_interval_steps}")
        
        # Load existing directory tree.
        if self.metric_dir.exists():
            self._existing_metric_dirs = [pathlib.Path(f.path) for f in os.scandir(self.metric_dir) if f.is_dir()]
            self._existing_metric_dirs = sorted(self._existing_metric_dirs, key=lambda p: int(p.stem)) # Ensure step sort order on start.
            self._latest_step_tracked = int(self._existing_metric_dirs[-1].stem)
        
        # Create parent directory tree if it does not already exist.
        else:
            self.metric_dir.mkdir(parents=True, exist_ok=True)
        
        # Build file name from directory stem.
        self._metric_filename = self.metric_dir.stem + "." + self.metric_file_type
    
    def append(self, metrics: dict):
        self._history.append(metrics)
    
    def extend(self, metrics: list[dict[str, Any]]):
        self._history.extend(metrics)
    
    def _save_history(self, step: int, history: list[dict[str, Any]]):
        """Perform save operation for given step."""
        # Create new sub-directory for current step.
        current_metric_dir = (self.metric_dir / str(step))
        
        # Check if directory already exists from a previous run.
        # If one exists, then delete it.
        if current_metric_dir.exists():
            shutil.rmtree(current_metric_dir)
            # Remove current metric dir from the list if it exists.
            try:
                self._existing_metric_dirs.remove(current_metric_dir)
            except:
                pass

        # Create directory for current save.
        current_metric_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current metrics to file within current step directory.
        filepath = current_metric_dir / self._metric_filename
        tools.save_train_results(filepath, history, **self.save_kwargs)
        
        # Preserve the current metric directory.
        self._existing_metric_dirs.append(current_metric_dir)
        # Preserve the current step as the latest step saved.
        self._latest_step_tracked = step
        
        # Remove old metrics if necessary.
        if self.max_to_keep is not None:
            while len(self._existing_metric_dirs) > self.max_to_keep:
                tobedeleted_metric_dir = self._existing_metric_dirs.pop(0) # Remove the first (oldest) element.
                try:
                    shutil.rmtree(tobedeleted_metric_dir)
                except Exception as e:
                    print(f"Error removing metric directory: {e}", file=sys.stderr)
    
    # TODO make saving async.
    def save(self, step: int, metrics: dict[str, Any]):
        """Save a given metric dictionary entry into the history list, and preserve as file according to save interval."""
        # Preserve data.
        self._history.append(metrics)
        self._latest_step_untracked = step
        
        # Save current data history to file at interval.
        if step % self.save_interval_steps == 0:
            self._save_history(step=step, history=self._history)
    
    def all_steps(self) -> list[int]:
        """Return all steps tracked by the manager."""
        steps = [int(d.stem) for d in self._existing_metric_dirs]
        steps = sorted(steps) # Ensure integer sort order.
        return steps

    def latest_step(self) -> int | None:
        """Returns the latest step saved."""
        return self._latest_step_tracked
    
    # TODO wait for async saves to finish.
    def wait_until_finished(self):
        # Manually save last entry if latest data was not tracked.
        # Only save if a save interval was specified.
        if self._latest_step_untracked is not None and self._latest_step_untracked != self._latest_step_tracked:
            self._save_history(step=self._latest_step_untracked, history=self._history)
    
    def load(self, step: int | None = None):
        # Load step directly.
        if step is not None:
            current_metric_dir = (self.metric_dir / str(step))
            if not current_metric_dir.exists():
                raise ValueError(f"Metrics for step {step} do not exist")
        # Load the latest step.
        else:
            current_metric_dir = (self.metric_dir / str(self.latest_step()))
        # Load the file.
        filepath = current_metric_dir / self._metric_filename
        self._history = tools.load_train_results(filepath)
        return self._history

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Remove hidden properties.
        d.pop('_history', None)
        d.pop('_existing_metric_dirs', None)
        d.pop('_metric_filename', None)
        d.pop('_latest_step_tracked', None)
        d.pop('_latest_step_untracked', None)
        return d