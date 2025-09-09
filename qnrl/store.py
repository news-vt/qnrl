"""This module is in charge of managing configuration storage via an SQLite3 database."""
from dataclasses import dataclass, field
import json
import uuid
import sqlite3
from filelock import SoftFileLock
import pathlib

from mashumaro import DataClassDictMixin

# Path of SQLite3 database file to store configuration JSON.
CONFIG_DB = 'config.db'


@dataclass(kw_only=True)
class ConfigStore(DataClassDictMixin):
    """Manage key-value storage of configurations within an SQLite3 database."""
    db: pathlib.Path = field(default=pathlib.Path(CONFIG_DB)) # Path to SQLite3 database.
    
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
            with sqlite3.connect(CONFIG_DB) as conn:
                cursor = conn.cursor()
                
                try:
                    # Ensure table exists.
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS configs (
                            uuid TEXT PRIMARY KEY,
                            json TEXT NOT NULL
                        )
                    """)
                    # Insert record or ignore if it already exists.
                    cursor.execute("""
                        INSERT OR IGNORE INTO configs (uuid, json) VALUES (?, ?)
                    """, (cfg_hash, cfg_str))
                    # Update the record contents to handle the case where it already existed. This allows records to be updated if they already exist.
                    cursor.execute("""
                        UPDATE configs SET json = ? WHERE uuid = ?
                    """, (cfg_str, cfg_hash))
                    conn.commit()
                except sqlite3.Error as e:
                    raise e

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
            with sqlite3.connect(CONFIG_DB) as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS configs (
                            uuid TEXT PRIMARY KEY,
                            json TEXT NOT NULL
                        )
                    """)
                    cursor.execute("""
                        SELECT * FROM configs WHERE uuid = ?
                    """, (cfg_hash,))
                    row = cursor.fetchone()
                    return row
                except sqlite3.Error as e:
                    raise e

    def get_all(self) -> list[tuple[str, str]]:
        """Retrieve all configuration JSON strings and associated unique UUID hashes from database."""
        with SoftFileLock(str(self.db) + ".lock", thread_local=True) as lock:
            with sqlite3.connect(CONFIG_DB) as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS configs (
                            uuid TEXT PRIMARY KEY,
                            json TEXT NOT NULL
                        )
                    """)
                    cursor.execute("""
                        SELECT * FROM configs
                    """)
                    rows = cursor.fetchall()
                    return rows
                except sqlite3.Error as e:
                    raise e
    
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