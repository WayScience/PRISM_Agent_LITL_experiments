"""
Connection lifecycle, canonical JSON, transactions and other small helpers
"""

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import sqlite3


def uid() -> str:
    return str(uuid4())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: object) -> str:
    """Canonical, finite JSON; NumPy scalars are accepted without importing NumPy."""

    def convert(item):
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Not JSON serializable: {type(item).__name__}")

    return json.dumps(
        value, sort_keys=True, allow_nan=False, default=convert, separators=(",", ":")
    )


def digest(value: object) -> str:
    return sha256(encode(value).encode()).hexdigest()


class SQLiteStore:
    """
    Manage a logger connection and provide private insert/lookup primitives.
    """

    def __init__(self, db_path: str | Path ="experiments.sqlite", *, timeout: float = 30):
        """
        Initialize the SQLiteStore with the given database path and timeout.

        :param db_path: Path to the SQLite database file.
        :param timeout: Timeout for SQLite connections in seconds.
        """

        if sqlite3.sqlite_version_info < (3, 37):
            raise RuntimeError("SQLite >=3.37 is required for STRICT tables")

        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=timeout, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            # Initialize before launching workers. Existing databases never run DDL.
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchone():
                    raise ValueError("Unrecognized database; use a new v2 file")
                schema = Path(__file__).with_name("schema.sql").read_text()
                self.connection.executescript(
                    "BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;"
                )
            elif version != 2:
                raise ValueError(
                    "Legacy/unsupported schema; retain it and use a new v2 file"
                )
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
        except BaseException:
            self.connection.close()
            raise

    @contextmanager
    def transaction(self):
        """
        Begin an immediate transaction for atomic operations.
        Commits on success, rolls back on exception.
        """
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def one(self, sql: str, parameters: tuple = ()):
        """
        Fetch a single row from the database.

        :param sql: SQL query to execute.
        :param parameters: Parameters for the SQL query.
        :return: A dictionary representing the row.
        :raises KeyError: If no row is found.
        """
        row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            raise KeyError(parameters)
        return dict(row)

    def rows(self, sql: str, parameters: tuple = ()):
        """
        Fetch multiple rows from the database.

        :param sql: SQL query to execute.
        :param parameters: Parameters for the SQL query.
        :return: A list of dictionaries representing the rows.
        """
        return [dict(row) for row in self.connection.execute(sql, parameters)]

    def insert(self, table: str, **values):
        """
        Insert a row into the specified table.

        :param table: Name of the table to insert into.
        :param values: Column-value pairs to insert.
        """

        columns = ",".join(values)
        marks = ",".join("?" for _ in values)
        self.connection.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values())
        )

    def close(self):
        """Close the database connection."""
        self.connection.close()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, *exc):
        """Exit the runtime context and close the database connection."""
        self.close()
