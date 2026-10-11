"""Small SQLite journal: artifacts are files, their identities and history live here."""

import json
import sqlite3
import time
import uuid
from pathlib import Path


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.db = sqlite3.connect(workspace / "state.db")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.listeners = []  # called with (kind, record) after each record write
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def meta(self, key, default=None):
        row = self.db.execute("SELECT payload FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, kind, record):
        with self.db:
            self.db.execute(
                "INSERT INTO records(kind,id,payload) VALUES (?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (kind, record["id"], json.dumps(record)),
            )
        for listener in self.listeners:
            listener(kind, record)

    def records(self, kind):
        rows = self.db.execute("SELECT payload FROM records WHERE kind=? ORDER BY seq", (kind,))
        return [json.loads(row[0]) for row in rows]

    def get(self, record_id):
        row = self.db.execute("SELECT payload FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise KeyError(record_id)
        return json.loads(row[0])

    def event(self, kind, **payload):
        with self.db:
            self.db.execute("INSERT INTO events(time,kind,payload) VALUES (?,?,?)",
                            (time.time(), kind, json.dumps(payload)))

    def close(self):
        self.db.close()

    def commit_expansion(self, expansion, candidates, stats, workflow_state=None):
        """Candidates, search statistics and workflow state change together, so a
        resumed run never sees one without the others."""
        with self.db:
            for candidate in candidates:
                self.db.execute("INSERT INTO records(kind,id,payload) VALUES ('candidate',?,?)",
                                (candidate["id"], json.dumps(candidate)))
            self.db.execute("UPDATE records SET payload=? WHERE id=?",
                            (json.dumps(expansion), expansion["id"]))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('search_stats',?)", (json.dumps(stats),))
            if workflow_state is not None:
                self.db.execute("INSERT OR REPLACE INTO meta VALUES ('workflow_state',?)",
                                (json.dumps(workflow_state),))
        for listener in self.listeners:
            for candidate in candidates:
                listener("candidate", candidate)
            listener("expansion", expansion)
