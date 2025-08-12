import json
import pathlib
import datetime
from typing import Callable

from flax import nnx
import orbax.checkpoint as ocp
from flax.training import orbax_utils
import jax.numpy as jnp
import numpy as np

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
    res = {k: jnp.array(v).tolist() for k,v in res.items()}
    return res


def save_train_results(filepath: str | pathlib.Path, d: dict):
    """Save training result dictionary to JSON file."""
    # Convert filepath to string and format with arguments.
    filepath = str(filepath).format(
        datetime=datetime.datetime.now().isoformat(),
    )
    with open(filepath, 'w+') as f:
        json.dump(d, f, cls=JAXJSONEncoder)


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