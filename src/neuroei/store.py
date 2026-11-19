import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .models import now

# Field names accepted by Store.select(); they are interpolated into a JSON path.
_FIELD = re.compile(r"^[A-Za-z0-9_]+$")


class Store:
    def __init__(self, root: Path, *, read_only=False):
        self.root = root
        self.path = root / "neuroei.sqlite3"
        self.read_only = read_only
        self._local = threading.local()
        self._json1 = None
        if read_only:
            if not self.path.is_file():
                raise ValueError("Existing database required for read-only inspection")
            return
        root.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS objects (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))")
            db.execute("CREATE TABLE IF NOT EXISTS audit (at TEXT, run_id TEXT, body TEXT)")

    def connect(self):
        """Open a new connection. The caller owns it and must close it when done."""
        if self.read_only:
            db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            db.execute("PRAGMA query_only=ON")
            return db
        return sqlite3.connect(self.path, timeout=10)

    def _connection(self):
        """One connection per thread: opening SQLite costs ~0.3 ms and drops the statement cache."""
        db = getattr(self._local, "db", None)
        if db is None:
            db = self.connect()
            self._local.db = db
        return db

    def close(self):
        """Close this thread's reused connection (the store stays usable afterwards)."""
        db = getattr(self._local, "db", None)
        if db is not None:
            self._local.db = None
            db.close()

    @contextmanager
    def transaction(self):
        db = self._connection()
        try:
            yield db
        except BaseException:
            db.rollback()
            raise
        else:
            db.commit()

    def put(self, kind: str, key: str, body: dict, *, immutable=False):
        encoded = json.dumps(body, ensure_ascii=False, sort_keys=True)
        with self.transaction() as db:
            old = db.execute("SELECT body FROM objects WHERE kind=? AND id=?", (kind, key)).fetchone()
            if immutable and old and old[0] != encoded:
                raise ValueError(f"Immutable {kind} already exists: {key}")
            db.execute("INSERT OR REPLACE INTO objects VALUES (?,?,?)", (kind, key, encoded))

    def get(self, kind: str, key: str):
        row = self._connection().execute(
            "SELECT body FROM objects WHERE kind=? AND id=?", (kind, key)).fetchone()
        if not row:
            raise ValueError(f"Missing {kind}: {key}")
        return json.loads(row[0])

    def list(self, kind: str):
        return [body for _, body in self.items(kind)]

    def items(self, kind: str):
        rows = self._connection().execute(
            "SELECT id, body FROM objects WHERE kind=? ORDER BY id", (kind,)).fetchall()
        return [(row[0], json.loads(row[1])) for row in rows]

    def select(self, kind: str, fields, *, key=None, include_id=False, decode=()):
        """Read only the requested top-level scalar fields for one kind.

        Used by first-screen reads that need a handful of metadata values from
        records whose bodies carry large payloads (report markdown, source
        blocks): the payload never crosses the SQLite boundary or json.loads.
        List/dict fields come back as JSON text unless named in `decode`.
        Falls back to a full parse on SQLite builds without the JSON1 extension.
        """
        fields = tuple(fields)
        if not fields or any(not _FIELD.match(field) for field in fields):
            raise ValueError("select() needs plain top-level field names")
        if self._json1_available():
            names = ("id", *fields) if include_id else fields
            columns = ", ".join((["id"] if include_id else [])
                                + [f"json_extract(body, '$.{field}')" for field in fields])
            sql = f"SELECT {columns} FROM objects WHERE kind=?"
            params = [kind]
            if key is not None:
                sql += " AND id=?"
                params.append(key)
            rows = self._connection().execute(sql + " ORDER BY id", params).fetchall()
            result = [dict(zip(names, row)) for row in rows]
            for item in result:
                for field in decode:
                    value = item.get(field)
                    if isinstance(value, str):
                        item[field] = json.loads(value)
            return result
        return [{**({"id": row_key} if include_id else {}),
                 **{field: body.get(field) for field in fields}}
                for row_key, body in self.items(kind) if key is None or row_key == key]

    def _json1_available(self):
        if self._json1 is None:
            try:
                self._connection().execute("SELECT json_extract('{}', '$.x')").fetchone()
                self._json1 = True
            except sqlite3.OperationalError:
                self._json1 = False
        return self._json1

    def delete(self, kind: str, key: str) -> int:
        with self.transaction() as db:
            cursor = db.execute("DELETE FROM objects WHERE kind=? AND id=?", (kind, key))
        return cursor.rowcount

    def audit(self, run_id: str, body: dict):
        # Callers supply metadata only: never prompts, reasoning, keys, or response bodies.
        with self.transaction() as db:
            db.execute("INSERT INTO audit VALUES (?,?,?)", (now(), run_id, json.dumps(body)))
