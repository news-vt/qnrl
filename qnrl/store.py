"""This module is in charge of managing configuration storage via an SQLite3 database."""
from dataclasses import dataclass, field
import json
import uuid
import sqlite3
from filelock import SoftFileLock
import pathlib

import sqlalchemy
import sqlalchemy.orm
from mashumaro import DataClassDictMixin


class TableBase(sqlalchemy.orm.DeclarativeBase):
    pass


class Config(TableBase):
    __tablename__ = "configs"
    
    uuid: sqlalchemy.orm.Mapped[str] = sqlalchemy.orm.mapped_column(primary_key=True)
    json: sqlalchemy.orm.Mapped[str]
    
    def __repr__(self) -> str:
        return f"Config(uuid={self.uuid!r}, json={self.json!r})"
    
    def to_dict(self) -> dict:
        return dict(uuid=self.uuid, json=self.json)
    
    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)


@dataclass(kw_only=True)
class ConfigStore(DataClassDictMixin):
    """Manage key-value storage of configurations within an under-the-hood SQLite3 database."""
    db: pathlib.Path = field(default=pathlib.Path('configs.db')) # Path to SQLite3 database.
    
    def __post_init__(self):
        # Store reference to database engine.
        self.engine = sqlalchemy.create_engine(f"sqlite:///{self.db}")
        
        # Create tables if they do not exist.
        TableBase.metadata.create_all(self.engine)
    
    def serialize_config(self, cfg: DataClassDictMixin) -> str:
        """Serialize a configuration object to JSON string format."""
        cfg_dict = cfg.to_dict()
        cfg_str = json.dumps(cfg_dict)
        return cfg_str

    def compute_config_hash(self, cfg: DataClassDictMixin) -> tuple[str, str]:
        """Generate a UUID hash for the given configuration object, and also return the string that was hashed.
        
        Hashes are generated as follows: `Config -> dictionary -> JSON -> UUIDv5`
        
        Returns `(cfg_hash, cfg_str)`.
        """
        cfg_str = self.serialize_config(cfg)
        cfg_hash = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=cfg_str)
        return str(cfg_hash), cfg_str

    def store_config_by_hash(self, cfg_hash: str, cfg_str: str):
        """Store configuration JSON string via unique UUID hash in database."""
        with SoftFileLock(str(self.db) + ".lock", thread_local=True) as lock:
            
            with sqlalchemy.orm.Session(self.engine) as session:
                cfg = Config(
                    uuid=cfg_hash,
                    json=cfg_str,
                )
                session.add(cfg)
                session.commit()

    def store_config(self, cfg: DataClassDictMixin):
        """Automatically generate unique hash for configuration and store it.
        
        Returns `(cfg_hash, cfg_str)`.
        """
        cfg_hash, cfg_str = self.compute_config_hash(cfg)
        self.store_config_by_hash(cfg_hash=cfg_hash, cfg_str=cfg_str)
        return cfg_hash, cfg_str

    def get_config_from_hash(self, cfg_hash: str) -> tuple[str, str]:
        """Retrieve configuration JSON string from unique UUID hash in database."""
        with SoftFileLock(str(self.db) + ".lock", thread_local=True) as lock:
            
            try:
                with sqlalchemy.orm.Session(self.engine) as session:
                    stmt = sqlalchemy.select(Config).where(Config.uuid == cfg_hash)
                    cfg = session.scalar(stmt)
                    return cfg.uuid, cfg.json
            except:
                raise ValueError(f"No entry found for hash {cfg_hash}")

    def get_all(self) -> list[tuple[str, str]]:
        """Retrieve all configuration JSON strings and associated unique UUID hashes from database."""
        with SoftFileLock(str(self.db) + ".lock", thread_local=True) as lock:
            
            with sqlalchemy.orm.Session(self.engine) as session:
                stmt = sqlalchemy.select(Config)
                rows = session.scalars(stmt).all()
                return [(cfg.uuid, cfg.json) for cfg in rows]
    
    def dumps(self) -> str:
        """Dump SQL database as JSON string."""
        objs = self.get_all()
        return json.dumps(objs)
    
    def dump(self) -> list[tuple[str, str]]:
        """Dump SQL database as JSON serializable object."""
        return json.loads(self.dumps())
    
    def __iter__(self):
        """Iterate over all items in database."""
        yield from self.get_all()
    
    def __getitem__(self, cfg_hash: str) -> str:
        """Retrieve configuration string from unique hash string like dictionary key."""
        try:
            _, cfg_str = self.get_config_from_hash(cfg_hash)
            return cfg_str
        except:
            raise KeyError(cfg_hash)
    
    def __setitem__(self, cfg_hash: str, cfg_str: str):
        """Store configuration string associated with unique has string."""
        self.store_config_by_hash(cfg_hash=cfg_hash, cfg_str=cfg_str)
    
    def __contains__(self, cfg_hash: str) -> bool:
        """Check if a configuration hash exists within the store."""
        keys, _ = list(zip(*self.get_all()))
        return cfg_hash in keys
    
    def __len__(self) -> int:
        """Get total number of elements in store."""
        return len(self.get_all())