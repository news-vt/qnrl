from dataclasses import dataclass
from typing import Any

from flax import nnx

from .serializable import SerializableModule

@dataclass(kw_only=True)
class Sequential(SerializableModule):
    """Sequential `nnx.Module` using either a `list[...]` or `dict[str, ...]` of modules.
    
    Always behaves like a list of elements, where the indices are either `int` (if layers are `list`) or `str` (if layers are `dict`).
    
    BEWARE of a quirk with `nnx.jit` where dictionary keys are automatically sorted. As a result, when passing a `dict[str, ...]` the key insertion order is not guaranteed on the `self.layers` member after calling `__call__()`. To remedy this, always use `__iter__()` to iterate over the key list. The key order will be preserved when serializing via `to_dict()`.
    
    If a `list[...]` is provided the PyTree will look like:
    >>> {
            'layers': {
                0: { ... },
                1: { ... },
                ...
            },
        }
    
    If a `dict[str, ...]` is provided, the original key order will be preserved via `__iter__()`, with the PyTree:
    >>> {
            'layers': {
                'key-0': { ... },
                'key-1': { ... },
                ...
            },
        }
    """
    layers: list[SerializableModule] | dict[str, SerializableModule]
    key_order: list[int] | list[str] # Private layer key ordering.
    
    def __init__(self, layers: list[SerializableModule] | dict[str, SerializableModule], key_order: list[int] | list[str] | None = None):
        self.layers = layers
        
        # Preserve the original layer order, because of a bug with `nnx.jit` automatically sorting dictionary keys.
        if key_order is None:
            self.key_order = list(self.layers.keys()) if isinstance(self.layers, dict) else list(range(len(self.layers)))
        else:
            self.key_order = key_order
    
    def __iter__(self):
        yield from (self.layers[key] for key in self.key_order)
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def __post_serialize__(self, d: dict[Any, Any]) -> dict[Any, Any]:
        # Get serialization of parent.
        d = super().__post_serialize__(d)
        # Preserve original input key order due to quirk of `nnx.jit`.
        if isinstance(d['layers'], dict):
            d['layers'] = {key: d['layers'][key] for key in self.key_order}
        return d
    
    @nnx.jit
    def __call__(self, x):
        for key in self.key_order:
            x = self.layers[key](x)
        return x