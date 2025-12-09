import uuid
import json

from mashumaro import DataClassDictMixin

type SerialType = int | float | str | bool | list | tuple | dict | None | DataClassDictMixin


def deserialize_object(obj_str: str, **kwargs) -> SerialType:
    """Decode an object from JSON string."""
    return json.loads(obj_str, **kwargs)


def serialize_object(obj: SerialType, **kwargs) -> str:
    """Serialize a object object to JSON string format."""
    # Standard types can be serialized ok.
    if isinstance(obj, (int | float | str | bool | list | tuple | dict | None)):
        obj_ser = obj
    # Ensure object can be serialized as a dictionary.
    else:
        assert hasattr(obj, 'to_dict'), "only standard Python types (bool, int, float, str, list, tuple, dict) or custom classes that implement a `to_dict()` method can be serialized."
        # De-serialize and re-serialize again to ensure types will be correct on load.
        # to_dict() -> from_dict() -> to_dict()
        obj_ser = obj.from_dict(obj.to_dict()).to_dict()
    # Generate a serialized object from the input using JSON.
    obj_str = json.dumps(obj_ser, **kwargs) # Dump serialized object as JSON string.
    return obj_str


def hash_serializable_object(obj: SerialType) -> tuple[str, str]:
    """Generate a UUID hash for the given object, and also return the string that was hashed.
    
    Hashes are generated as follows: `Config -> dictionary -> JSON -> UUIDv5`
    
    Returns `(obj_hash, obj_str)`.
    """
    obj_str = serialize_object(obj)
    obj_hash = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=obj_str)
    return str(obj_hash), obj_str