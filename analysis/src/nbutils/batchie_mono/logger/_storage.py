"""Own SQLite resources and transaction boundaries for the logger package.

This module depends only on the standard library. ``SQLiteStore`` creates new
databases from the adjacent SQL schema, checks existing database version/table
names, enables foreign keys, and ensures child-record lookup indexes exist.
It does not interpret traces, JSON payloads, or experiment semantics.

Each ``_insert`` is its own connection transaction. Closing the store does not
undo previous successful writes, and the context manager does not wrap a whole
experiment in a transaction. Connections retain SQLite's same-thread restriction;
use a controller writer rather than sharing a logger across sampling workers.
"""

import sqlite3
from contextlib import AbstractContextManager
from pathlib import Path


class SQLiteStore(AbstractContextManager):
    """Manage a logger connection and provide private insert/lookup primitives.

    ``connection`` is a sqlite3 connection using named ``sqlite3.Row`` results.
    Context entry returns this instance; context exit closes it without
    suppressing exceptions. Subclasses supply artifact-specific operations.
    """

    def __init__(self, db_path="experiments.sqlite"):
        """Open a database and initialize or check its version-1 schema.

        Args:
            db_path: Filesystem path accepted by pathlib.Path. Missing parent
                directories are created; SQLite waits up to 30 seconds for locks.

        Raises:
            ValueError: An existing database has an unsupported version or
                lacks a required table. Validation checks version and table
                names, not every column definition.

        Empty unversioned databases are initialized atomically from schema.sql.
        Existing v1 databases gain missing lookup indexes without a version
        change. Filesystem/SQLite errors propagate; if setup fails after opening
        the connection, it is closed before the exception is re-raised.
        """
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if version == 0 and not tables:
                schema = Path(__file__).with_name("schema.sql").read_text()
                self.connection.executescript("BEGIN;\n" + schema + "\nCOMMIT;")
            elif version != 1 or not {
                "experiment_runs",
                "predictions",
                "batches",
            }.issubset(tables):
                raise ValueError("Unsupported experiment logger schema")
            # Also upgrade query performance for existing v1 databases.
            with self.connection:
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS predictions_by_run ON predictions(run_id)"
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS batches_by_run ON batches(run_id)"
                )
        except Exception:
            self.close()
            raise


    def close(self):
        """Close the owned connection, releasing database resources.

        Earlier committed records remain persisted. Pending caller-managed
        transactions are not committed by this method. Repeated closes are safe;
        subsequent queries or writes on the closed connection fail.
        """
        self.connection.close()


    def __exit__(self, *args):
        """Close on context exit, returning None so exceptions propagate.

        Args:
            *args: Exception type, value, and traceback from the with statement;
                accepted for the context-manager protocol, otherwise unused.
        """
        self.close()


    def _insert(self, sql, values):
        """Execute one parameterized write and return its last inserted row ID.

        Args:
            sql: Internal SQL statement using SQLite parameter placeholders.
            values: Bound parameter values, never interpolated into SQL text.

        Returns:
            int or None: SQLite cursor.lastrowid (an integer for current inserts).

        The connection context commits on success and rolls back on failure;
        database errors propagate. Callers must not use this helper inside a
        larger transaction expecting several logger calls to be atomic: the
        context also commits any transaction already active on this connection.
        """
        with self.connection:
            return self.connection.execute(sql, values).lastrowid


    def _row(self, sql, key):
        """Fetch one keyed row as a mutable dictionary without committing.

        Args:
            sql: Internal lookup statement with one bound parameter.
            key: Value bound to that parameter.

        Returns:
            dict: First matching row with column names as keys.

        Raises:
            KeyError: No row matches ``key``. SQLite errors propagate unchanged.
        """
        row = self.connection.execute(sql, (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return dict(row)
