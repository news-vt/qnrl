"""This module is in charge of managing object storage via an SQLite3 database."""
from dataclasses import dataclass, field
import json
import uuid
from filelock import SoftFileLock
import pathlib

from mashumaro import DataClassDictMixin


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
    
    def serialize(self, obj: int | float | str | bool | list | tuple | dict | DataClassDictMixin) -> str:
        """Serialize a object object to JSON string format."""
        # Standard types can be serialized ok.
        if isinstance(obj, (bool, int, float, str, list, tuple, dict)):
            obj_ser = obj
        # Ensure object can be serialized as a dictionary.
        else:
            assert hasattr(obj, 'to_dict'), "only standard Python types (bool, int, float, str, list, tuple, dict) or custom classes that implement a `to_dict()` method can be serialized."
            # Serialize object as dictionary.
            obj_ser = obj.to_dict()
        # Dump serialized object as JSON string.
        obj_str = json.dumps(obj_ser)
        return obj_str

    def compute_hash(self, obj: int | float | str | bool | list | tuple | dict | DataClassDictMixin) -> tuple[str, str]:
        """Generate a UUID hash for the given object, and also return the string that was hashed.
        
        Hashes are generated as follows: `Config -> dictionary -> JSON -> UUIDv5`
        
        Returns `(obj_hash, obj_str)`.
        """
        obj_str = self.serialize(obj)
        obj_hash = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=obj_str)
        return str(obj_hash), obj_str

    @staticmethod
    def load(path: pathlib.Path | str) -> dict[str, str]:
        """Load contents of config storage file designated by JSON file path."""
        # Ensure path is a true `pathlib.Path` object.
        path = pathlib.Path(path)
        # Ensure file path is JSON.
        assert path.suffix == '.json', "store file path must have extension `.json`"
        # Lock file access.
        with SoftFileLock(str(path) + ".lock", thread_local=True) as lock:
            # Open the file for reading.
            with open(path, 'r') as f:
                # Decode file contents as JSON data.
                return json.load(f)
    
    @staticmethod
    def store(path: pathlib.Path | str, obj_hash: str, obj_str: str) -> tuple[str, str]:
        """Record serialized object JSON string via unique UUID hash in store designated by JSON file path."""
        # Ensure path is a true `pathlib.Path` object.
        path = pathlib.Path(path)
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

    def store_obj_by_hash(self, obj_hash: str, obj_str: str) -> tuple[str, str]:
        """Record serialized object JSON string via unique UUID hash in store."""
        return self.store(path=self.path, obj_hash=obj_hash, obj_str=obj_str)

    def store_obj(self, obj: DataClassDictMixin) -> tuple[str, str]:
        """Automatically generate unique hash for object and store it.
        
        Returns `(obj_hash, obj_str)`.
        """
        obj_hash, obj_str = self.compute_hash(obj)
        return self.store_obj_by_hash(obj_hash=obj_hash, obj_str=obj_str)

    def get_config_from_hash(self, obj_hash: str) -> str:
        """Retrieve object JSON string from unique UUID hash in store."""
        objs = self.load(path=self.path)
        return objs[obj_hash]

    def get_all(self) -> dict[str, str]:
        """Retrieve all object JSON strings and associated unique UUID hashes from store."""
        return self.load(path=self.path)
    
    def exists(self, hash_or_obj: str | DataClassDictMixin) -> bool:
        """Check if an object or hash exists in store.
        
        If `str` then assume it is a hash, if `DataClassDictMixin` treat it as a raw object, otherwise throw error.
        """
        if isinstance(hash_or_obj, DataClassDictMixin):
            obj_hash, _ = self.compute_hash(hash_or_obj)
        elif isinstance(hash_or_obj, str):
            obj_hash = hash_or_obj
        else:
            raise ValueError(f"unsupported type for object {hash_or_obj}")
        objs = self.load(path=self.path)
        return obj_hash in objs
    
    def dumps(self) -> str:
        """Dump store as JSON string."""
        objs = self.load(path=self.path)
        return json.dumps(objs)
    
    def dump(self) -> list[tuple[str, str]]:
        """Dump store as JSON serializable object."""
        return json.loads(self.dumps())
    
    def get(self, obj_hash: str, default = None):
        """Safe retrieval with default if no key is present."""
        try:
            return self.get_config_from_hash(obj_hash)
        except:
            return default
    
    def items(self):
        """Get hash store items as alias of `dict.items()`."""
        return self.get_all().items()
    
    def values(self):
        """Get hash store values as alias of `dict.values()`."""
        return self.get_all().values()
    
    def keys(self):
        """Get hash store keys as alias of `dict.keys()`."""
        return self.get_all().keys()
    
    def __iter__(self):
        """Iterate over all serialized objects and their hashes in database."""
        yield from self.get_all()
    
    def __getitem__(self, obj_hash: str) -> str:
        """Retrieve serialized object string from unique hash string like dictionary key."""
        try:
            return self.get_config_from_hash(obj_hash)
        except:
            raise KeyError(obj_hash)
    
    def __setitem__(self, obj_hash: str, obj: str | DataClassDictMixin):
        """Store object -- either as string or serializable object -- associated with unique hash string."""
        if isinstance(obj, DataClassDictMixin):
            obj_str = self.serialize(obj)
        else:
            obj_str = obj
        self.store_obj_by_hash(obj_hash=obj_hash, obj_str=obj_str)
    
    def __contains__(self, hash_or_obj: str | DataClassDictMixin) -> bool:
        """Check if an object or hash exists within the store.
        
        Alias of `exists()`.
        """
        return self.exists(hash_or_obj=hash_or_obj)
    
    def __len__(self) -> int:
        """Get total number of elements in store."""
        return len(self.get_all())