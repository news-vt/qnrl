import json

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