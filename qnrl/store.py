"""This module is in charge of managing object storage via an SQLite3 database."""
from dataclasses import dataclass, field
import json
import uuid
from filelock import SoftFileLock
import pathlib

from mashumaro import DataClassDictMixin


type SerialType = int | float | str | bool | list | tuple | dict | None


@dataclass(kw_only=True)
class HashStore(DataClassDictMixin):
    """Key-value storage of serializable objects (i.e., must implement `.to_dict()`).
    
    Data is stored in a JSON file under the hood.
    
    This class is designed to be used like a traditional dictionary, without knowledge of the underlying mechanism used for storage.
    """
    path: pathlib.Path | str = field(default=pathlib.Path('hashstore.json'))
    
    def __post_init__(self):
        # Ensure path is a true `pathlib.Path` object.
        self.path = pathlib.Path(self.path)
        
        # Ensure file path is JSON.
        assert self.path.suffix == '.json', "store file path must have extension `.json`"
        
        # Create file if it does not exist.
        if not self.path.exists():
            with open(self.path, 'w+') as f:
                json.dump({}, f)
    
    def encode(self, obj: SerialType | DataClassDictMixin, **kwargs) -> str:
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
    
    def decode(self, obj_str: str, **kwargs) -> SerialType:
        """Deserialize an object from JSON string."""
        return json.loads(obj_str, **kwargs)

    def hash(self, obj: SerialType | DataClassDictMixin) -> tuple[str, str]:
        """Generate a UUID hash for the given object, and also return the string that was hashed.
        
        Hashes are generated as follows: `Config -> dictionary -> JSON -> UUIDv5`
        
        Returns `(obj_hash, obj_str)`.
        """
        obj_str = self.encode(obj)
        obj_hash = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=obj_str)
        return str(obj_hash), obj_str

    def load(self) -> dict[str, str]:
        """Retrieve all object JSON strings and associated unique UUID hashes from store."""
        # Ensure path is a true `pathlib.Path` object.
        path = pathlib.Path(self.path)
        # Ensure file path is JSON.
        assert path.suffix == '.json', "store file path must have extension `.json`"
        # Lock file access.
        with SoftFileLock(str(path) + ".lock", thread_local=True) as lock:
            # Open the file for reading.
            with open(path, 'r') as f:
                # Decode file contents as JSON data.
                return json.load(f)
    
    def store(self, obj_hash: str, obj_str: str) -> tuple[str, str]:
        """Record serialized object JSON string via unique UUID hash in store designated by JSON file path."""
        # Ensure object to store is a string.
        assert isinstance(obj_str, str), "only strings can be stored"
        # Ensure path is a true `pathlib.Path` object.
        path = pathlib.Path(self.path)
        # Ensure file path is JSON.
        assert path.suffix == '.json', "store file path must have extension `.json`"
        # Lock file access.
        with SoftFileLock(str(path) + ".lock", thread_local=True) as lock:
            # Load existing file contents as JSON data.
            with open(path, 'r') as f:
                objs = json.load(f)
            # Add current entry.
            objs[obj_hash] = obj_str
            # Open the file for writing.
            with open(path, 'w+') as f:
                # Dump object to file as JSON data.
                json.dump(objs, f)
                # Return the hash and serialized string.
                return obj_hash, obj_str
    
    def store_obj(self, obj: DataClassDictMixin) -> tuple[str, str]:
        """Automatically generate unique hash for object and store it.
        
        Returns `(obj_hash, obj_str)`.
        """
        obj_hash, obj_str = self.hash(obj)
        return self.store(obj_hash=obj_hash, obj_str=obj_str)
    
    def remove(self, obj_hash: str, default = None) -> tuple[str, str]:
        """Remove a serialized object via unique UUID hash in store designated by JSON file path."""
        # Ensure path is a true `pathlib.Path` object.
        path = pathlib.Path(self.path)
        # Ensure file path is JSON.
        assert path.suffix == '.json', "store file path must have extension `.json`"
        # Lock file access.
        with SoftFileLock(str(path) + ".lock", thread_local=True) as lock:
            # Load existing file contents as JSON data.
            with open(path, 'r') as f:
                objs = json.load(f)
            # Remove designated entry if it exists.
            obj_str = objs.pop(obj_hash, default)
            # Open the file for writing.
            with open(path, 'w+') as f:
                # Dump object to file as JSON data.
                json.dump(objs, f)
                # Return the hash and serialized string.
                return obj_hash, obj_str
    
    def exists(self, hash_or_obj: str | DataClassDictMixin) -> bool:
        """Check if an object or hash exists in store.
        
        If `str` then assume it is a hash, if `DataClassDictMixin` treat it as a raw object, otherwise throw error.
        """
        if isinstance(hash_or_obj, DataClassDictMixin):
            obj_hash, _ = self.hash(hash_or_obj)
        elif isinstance(hash_or_obj, str):
            obj_hash = hash_or_obj
        else:
            raise ValueError(f"unsupported type for object {hash_or_obj}")
        objs = self.load()
        return obj_hash in objs
    
    def dumps(self) -> str:
        """Dump store as JSON string."""
        objs = self.load()
        return json.dumps(objs)
    
    def dump(self) -> list[tuple[str, str]]:
        """Dump store as JSON serializable object."""
        return json.loads(self.dumps())
    
    def get(self, obj_hash: str, default = None):
        """Retrieve object JSON string from unique UUID hash in store, with default if not key is present."""
        try:
            objs = self.load()
            return objs[obj_hash]
        except:
            return default
    
    def items(self):
        """Get hash store items as alias of `dict.items()`."""
        return self.load().items()
    
    def values(self):
        """Get hash store values as alias of `dict.values()`."""
        return self.load().values()
    
    def keys(self):
        """Get hash store keys as alias of `dict.keys()`."""
        return self.load().keys()
    
    def pop(self, obj_hash: str, default = None):
        """Remove object associated with unique hash string and return the object if it exists, or return the specified default otherwise."""
        try:
            _, cfg_str = self.remove(obj_hash=obj_hash, default=default)
            return cfg_str
        except:
            return default
    
    def __iter__(self):
        """Iterate over all serialized objects and their hashes in database."""
        yield from self.load()
    
    def __getitem__(self, obj_hash: str) -> str:
        """Retrieve serialized object string from unique hash string like dictionary key."""
        try:
            objs = self.load()
            return objs[obj_hash]
        except:
            raise KeyError(obj_hash)
    
    def __delitem__(self, obj_hash: str):
        """Remove object associated with unique hash string."""
        self.remove(obj_hash=obj_hash)
    
    def __setitem__(self, obj_hash: str, obj: str | DataClassDictMixin):
        """Store object -- either as string or serializable object -- associated with unique hash string."""
        if isinstance(obj, DataClassDictMixin):
            obj_str = self.encode(obj)
        else:
            obj_str = obj
        self.store(obj_hash=obj_hash, obj_str=obj_str)
    
    def __contains__(self, hash_or_obj: str | DataClassDictMixin) -> bool:
        """Check if an object or hash exists within the store.
        
        Alias of `exists()`.
        """
        return self.exists(hash_or_obj=hash_or_obj)
    
    def __len__(self) -> int:
        """Get total number of elements in store."""
        return len(self.load())
    
    def __repr__(self) -> str:
        """Represent the store as a dictionary string."""
        objs = self.load()
        return repr(objs)