"""This module is in charge of managing configuration storage via an SQLite3 database."""

import json
import uuid
import sqlite3

from mashumaro import DataClassDictMixin

# Path of SQLite3 database file to store configuration JSON.
CONFIG_DB = 'config.db'

def compute_config_hash(cfg: DataClassDictMixin) -> tuple[str, str]:
    """Generate a UUID hash for the given configuration object, and also return the string that was hashed.
    
    Hashes are generated as follows:
    
    `Config -> dictionary -> JSON -> UUIDv5`
    """
    cfg_dict = cfg.to_dict()
    cfg_str = json.dumps(cfg_dict)
    cfg_hash = uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=cfg_str)
    return str(cfg_hash), cfg_str


def db_store_config_by_hash(cfg_hash: str, cfg_str: str):
    """Store configuration JSON string via unique UUID hash in database."""
    # TODO add lock file.
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
                INSERT OR IGNORE INTO configs (uuid, json) VALUES (?, ?)
            """, (cfg_hash, cfg_str))
            conn.commit()
        except sqlite3.Error as e:
            raise e


def db_get_config_from_hash(cfg_hash: str) -> tuple[str, str]:
    """Retrieve configuration JSON string from unique UUID hash in database."""
    # TODO add lock file.
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

def db_getall() -> list[tuple[str, str]]:
    """Retrieve all configuration JSON strings and associated unique UUID hashes from database."""
    # TODO add lock file.
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