from dataclasses import dataclass
import json
import pathlib
import datetime
import importlib.util
import types
from typing import Any, Callable, Literal, TypeVar
import gzip
import os
import time

from flax import nnx
import jax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import jax.numpy as jnp
import numpy as np
import yaml
from mashumaro import DataClassDictMixin
import scipy.linalg

# Generic type for Flax NNX module.
M = TypeVar('M', bound=nnx.Module)

class NumpyJSONEncoder(json.JSONEncoder):
    """Serializes numpy objects for JSON support."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class JAXJSONEncoder(json.JSONEncoder):
    """Serializes JAX objects for JSON support."""
    def default(self, obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        elif isinstance(obj, jnp.floating):
            return float(obj)
        elif isinstance(obj, jnp.integer):
            return int(obj)
        return super().default(obj)


class DatetimeJSONEncoder(json.JSONEncoder):
    """Serializes `datetime.datetime` objects."""
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)


def make_combined_json_encoder(*encoder_classes: type[json.JSONEncoder]):
    class _Combined(json.JSONEncoder):
        def default(self, obj):
            for enc_cls in encoder_classes:
                try:
                    return enc_cls().default(obj)
                except TypeError:
                    pass
            return super().default(obj)
    return _Combined


def listofdict_to_dictoflist(listofdict: list[dict]) -> dict[str,list]:
    """Convert a list of dictionaries into dictionary of lists.
    
    Example:
    >>> ld = [{'a': 1, 'b': 2, 'c': [3,4]}, {'a': 5, 'b': 6, 'c': [7, 8]}]
    >>> dl = listofdict_to_dictoflist(ld)
    >>> dl['a'] # [1, 5]
    >>> dl['b'] # [2, 6]
    >>> dl['c'] # [[3,4], [7,8]]
    """
    res = {}
    for d in listofdict:
        for k,v in d.items():
            res.setdefault(k, []).append(v)
    res = {k: np.array(v).tolist() for k,v in res.items()}
    return res

@dataclass(kw_only=True)
class MetricStatistics(DataClassDictMixin):
    key: str # The unique identifier for the metric.
    shape: list[int]
    dtype: str
    min: float
    max: float
    argmin: int
    argmax: int
    mean: float
    std: float

    @classmethod
    def compute(cls, key: str, data: list):
        return cls(
            key=key,
            shape=list(jnp.array(data).shape),
            dtype=str(jnp.array(data).dtype),
            min=jnp.array(data).min().item(),
            max=jnp.array(data).max().item(),
            argmin=jnp.array(data).argmin().item(),
            argmax=jnp.array(data).argmax().item(),
            mean=jnp.array(data).mean().item(),
            std=jnp.array(data).std().item(),
        )


def compute_train_result_statistics(d: dict[str,list]) -> list[MetricStatistics]:
    """Computes statistics for metrics specified as a keyed dictionary of lists.
    
    Example input:
    >>> d = {'metric0': [1,2,3], 'metric1': [4,5,6,7]}
    >>> compute_train_result_statistics(d)
    """
    return [
        MetricStatistics.compute(key=metric_key, data=val)
        for metric_key, val in d.items()
    ]


def save_train_results(filepath: str | pathlib.Path, obj: Any, **kwargs):
    """Save training result dictionary to file.
    
    Supported file formats:
    - `'.json'`: JSON.
    - `'.json.gz'`: Compressed JSON using `gzip`.
    - `'.yaml' | '.yml'`: YAML.
    - `'.yaml.gz' | '.yml.gz'`: Compressed YAML using `gzip`.
    - `'.npy'`: Numpy via `jnp.save()`.
    - `'.npz'`: Numpy via `jnp.savez()` if `compressed=False`, or `np.savez_compressed()` if `compressed=True`.
    """
    # Convert filepath to string and format with arguments.
    filepath = pathlib.Path(str(filepath).format(
        datetime=datetime.datetime.now().isoformat(),
    ))
    # Check file extension type.
    match filepath.suffixes:
        case [] | ['.json'] | ['.json', '.gz']:
            if '.gz' in filepath.suffixes:
                with gzip.open(str(filepath), 'wt+', encoding="utf-8") as f:
                    kwargs_default = dict(cls=make_combined_json_encoder(NumpyJSONEncoder, JAXJSONEncoder))
                    json.dump(obj, f, **{**kwargs_default, **kwargs})
            else:
                with open(str(filepath), 'w+') as f:
                    kwargs_default = dict(cls=make_combined_json_encoder(NumpyJSONEncoder, JAXJSONEncoder))
                    json.dump(obj, f, **{**kwargs_default, **kwargs})
        case ['.yaml'] | ['.yml'] | ['.yaml', '.gz'] | ['.yml', '.gz']:
            if '.gz' in filepath.suffixes:
                with gzip.open(str(filepath), 'wt+', encoding="utf-8") as f:
                    yaml.dump(obj, f, **kwargs)
            else:
                with open(str(filepath), 'w+') as f:
                    yaml.dump(obj, f, **kwargs)
        case ['.npy']:
            jnp.save(str(filepath), obj, allow_pickle=True)
        case ['.npz']:
            if kwargs.get('compressed', False):
                np.savez_compressed(str(filepath), obj, allow_pickle=True) # No JAX variant of `jnp.savez_compressed()`.
            else:
                jnp.savez(str(filepath), obj, allow_pickle=True)
        case _:
            raise ValueError(f"{filepath.suffix} is an unsupported metric file type.")


def load_train_results(filepath: str | pathlib.Path) -> Any:
    """Load training result dictionary from file.
    
    Supported file formats:
    - `'.json'`: JSON.
    - `'.json.gz'`: Compressed JSON using `gzip`.
    - `'.yaml' | '.yml'`: YAML.
    - `'.yaml.gz' | '.yml.gz'`: Compressed YAML using `gzip`.
    - `'.npy' | '.npz'`: Numpy. Returned result will be a dictionary.
    """
    filepath = pathlib.Path(filepath)
    
    # Check file extension type.
    match filepath.suffixes:
        case [] | ['.json'] | ['.json', '.gz']:
            if '.gz' in filepath.suffixes:
                with gzip.open(str(filepath), 'rt') as f:
                    return json.load(f)
            else:
                with open(str(filepath), 'r') as f:
                    return json.load(f)
        case ['.yaml'] | ['.yml'] | ['.yaml', '.gz'] | ['.yml', '.gz']:
            if '.gz' in filepath.suffixes:
                with gzip.open(str(filepath), 'rt') as f:
                    return yaml.safe_load(f)
            else:
                with open(str(filepath), 'r') as f:
                    return yaml.safe_load(f)
        case ['.npy'] | ['.npz']:
            return jnp.load(str(filepath), allow_pickle=True)
        case _:
            raise ValueError(f"{filepath.suffix} is an unsupported metric file type.")


def serialize_train_results(obj: Any, type: Literal['json', 'yaml', 'yml'], **kwargs) -> str:
    """Serialize training results to an encoded string in format `['json', 'yaml', 'yml']`."""
    match type:
        case 'json':
            kwargs_default = dict(cls=make_combined_json_encoder(NumpyJSONEncoder, JAXJSONEncoder))
            return json.dumps(obj, **{**kwargs_default, **kwargs})
        case 'yaml' | 'yml':
            kwargs_default = dict(sort_keys=False, default_flow_style=False, indent=2)
            return yaml.dump(obj, **{**kwargs_default, **kwargs})
        case _:
            raise ValueError(f"{type} is an unsupported metric file type.")


def save_model(path: str | pathlib.Path, model: nnx.Module):
    """Save Flax NNX model."""
    state = nnx.state(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(f"{path}/state", state, force=True)
    checkpointer.wait_until_finished()


def restore_model_from_path(path: str | pathlib.Path, model: M) -> M:
    """Restore a given Flax NNX model to the state designed at the given checkpoint path."""
    state = nnx.state(model) # Get the state of the abstract model.
    # Load state from the saved model.
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(f"{path}/state", state)
    checkpointer.wait_until_finished()
    # Update the abstract model with the loaded state.
    nnx.update(model, state)
    return model


def load_model_from_path(path: str | pathlib.Path, model_create_func: Callable[[], M]) -> M:
    """Load a Flax NNX model from a given checkpoint path."""
    model = nnx.eval_shape(model_create_func) # Create the model with abstract shapes.
    return restore_model_from_path(path, model) # Restore from checkpoint path state.


def restore_model_from_checkpointmanager(
    model: M,
    checkpoint_manager: ocp.CheckpointManager,
    step: int | None = None, # Step to restore (defaults to latest step).
    ) -> M:
    """Restore model parameters from latest checkpoint (if one exists) from a given `orbax.checkpoint.CheckpointManager` instance."""
    # Restore model state from checkpoint.
    original_state = nnx.state(model)
    # Get latest step if none was provided.
    if step is None:
        step = checkpoint_manager.latest_step()
    restored_state = checkpoint_manager.restore(step, items=original_state)
    nnx.update(model, restored_state)
    # Return the model object.
    return model


def get_model_total_parameter_count(model: nnx.Module) -> int:
    """Return the total number of parameters within a Flax NNX module."""
    params = nnx.state(model, nnx.Param)
    return sum(jnp.prod(jnp.array(x.shape, dtype=int)) for x in jax.tree.leaves(params)).item()


def load_python_module_from_filepath(path: pathlib.Path) -> types.ModuleType:
    """Load a Python module from filepath."""
    mod_name = f"{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def inspect_flax_nnx_model(model: nnx.Module, include_raw_parameters: bool = False) -> dict[str, Any]:
    """Get details of Flax NNX model in serializable dictionary format."""
    params = nnx.state(model, nnx.Param)
    total_params = int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))
    info = jax.tree_util.tree_map(lambda p: dict(shape=p.shape, dtype=str(p.dtype)), params)
    d = {
        'parameters': {
            'total': total_params, # Total number of parameters.
            'info': nnx.to_pure_dict(info), # Model parameter information in PyTree format.
        },
    }
    if include_raw_parameters:
        d['parameters']['param'] = nnx.to_pure_dict(params) # Model parameters in PyTree format.
    return d


def wait_for_file_change(path: pathlib.Path, interval: float = 1.0) -> bool:
    """Blocking wait for file change. Returns `True` on file change, and `False` on error."""
    orig_mtime = os.path.getmtime(path)
    try:
        while True:
            if os.path.getmtime(path) > orig_mtime:
                return True
            time.sleep(interval) # Wait for next check.
    except:
        return False

def wait_for_list_of_file_change(paths: list[pathlib.Path], interval: float = 1.0) -> bool:
    """Blocking wait for multiple file changes. Returns `True` on any file change, and `False` on error."""
    orig_mtimes = [os.path.getmtime(p) for p in paths]
    try:
        while True:
            # Check all modified times.
            for path, orig_mtime in zip(paths, orig_mtimes):
                if os.path.getmtime(path) > orig_mtime:
                    return True
            time.sleep(interval) # Wait for next check.
    except:
        return False

def density_matrix(state: jax.Array) -> jax.Array:
    """Construct a density matrix from a given state vector."""
    rho = jnp.outer(state, jnp.conj(state).T) # |psi><psi|
    return rho

def is_unitary(m: jax.Array, **kwargs):
    """Check if a given matrix is unitary."""
    return jnp.allclose(jnp.eye(m.shape[0], dtype=m.dtype), m @ m.conj().T, **kwargs)

def is_hermitian(m: jax.Array, **kwargs):
    """Check if a given matrix is Hermitian."""
    return jnp.allclose(m, m.conj().T, **kwargs)

def is_diagonal(m: jax.Array, tol: float = 1e-5):
    """Check if a given matrix is diagonal (i.e., zero everywhere other than the diagonal elements).
    
    The input matrix is assumed to be square with shape `M x M`.
    """
    # Ensure matrix is square.
    i, j = m.shape
    assert i == j
    # Remove diagonal elements.
    diag_removed = m - jnp.diag(jnp.diag(m))
    # Check if any elements are greater than tolerance (i.e., more than zero).
    return ~jnp.any(diag_removed > tol)

def is_pure(m: jax.Array, **kwargs):
    """Check if a density matrix is pure by $Tr(m^2) = Tr(m)$."""
    return jnp.allclose(jnp.trace(m @ m.conj().T), jnp.trace(m), **kwargs)

def purity(m: jax.Array):
    r"""Compute the purity of a density matrix by $Tr(m^2) \in [0, 1]$.
    
    A density matrix is pure if $Tr(m^2) = 1$, and is mixed if $Tr(m^2) < 1$.
    """
    return jnp.trace(m @ m.conj().T)

def vn_entropy(rho: jax.Array, *, base: float | None = None) -> jax.Array:
    r"""Compute the von Neumann entropy of a density matrix rho by: $S(\rho) = -Tr(\rho \log_{2}(\rho))$
    
    Uses the natural logarithm by default. Otherwise, uses the base of the logarithm specified (e.g., `base=2`).
    """
    assert len(rho.shape) == 2 and rho.shape[0] == rho.shape[1], "density matrix must be square"
    
    # Natural log.
    if base is None:
        log_base = 1.
    # All other logarithms.
    else:
        log_base = jnp.log(base)
    
    def compute_eigenvalues(m: jax.Array) -> jax.Array:
        # Compute eigenvalues for matrix.
        eigenvalues = jnp.linalg.eigvalsh(rho)
        # Filter out zero eigenvalues (so that the logarithm is defined).
        eigenvalues: jax.Array = jnp.where(eigenvalues > 0., eigenvalues, 1.)
        return eigenvalues

    eigenvalues_rho = compute_eigenvalues(rho)
    entropy = -(eigenvalues_rho @ (jnp.log(eigenvalues_rho) / log_base))
    return entropy