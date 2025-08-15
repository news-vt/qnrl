from dataclasses import dataclass, field
import pathlib
import os
from typing import Any

from mashumaro.mixins.dict import DataClassDictMixin

from . import tools


@dataclass(kw_only=True)
class MetricManager(DataClassDictMixin):
    path: pathlib.Path
    save_kwargs: dict = field(default_factory=dict)
    save_frequency: int | None = None
    
    _history: list[dict[str, Any]] = field(default_factory=list, init=False)
    
    @property
    def history(self) -> list[dict[str, Any]]:
        return self._history
    
    @property
    def metrics(self) -> dict[str, Any]:
        return tools.listofdict_to_dictoflist(self._history) # list[dict] -> dict[str, Any]
    
    def __post_init__(self):
        # Create parent directory tree if it does not already exist.
        os.makedirs(self.path.parent, exist_ok=True)
    
    def append(self, metrics: dict):
        self._history.append(metrics)
    
    def extend(self, metrics: list[dict[str, Any]]):
        self._history.extend(metrics)
    
    # TODO make saving async.
    def save(self, step: int, metrics: dict[str, Any]):
        self._history.append(metrics)
        
        # Save to file at interval.
        if self.save_frequency is not None and step % self.save_frequency == 0:
            tools.save_train_results(self.path, self._history, **self.save_kwargs)
    
    # TODO wait for async saves to finish.
    def wait_until_finished(self):
        # Run final save of any metrics in memory.
        # metrics = tools.listofdict_to_dictoflist(self._history) # list[dict] -> dict[str, Any]
        tools.save_train_results(self.path, self._history, **self.save_kwargs)
    
    def load(self):
        self._history = tools.load_train_results(self.path)
        return self._history

    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Remove hidden properties.
        d.pop('_history', None)
        return d