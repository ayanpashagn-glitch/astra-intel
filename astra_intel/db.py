"""SQLite persistence: documents, page-bound chunks, chat history."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("ASTRA_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "astra_intel.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  sha256 TEXT NOT NULL UNIQUE,
  pages INTEGER NOT NULL,
  n_chunks INTEGER NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  summary_mode TEXT NOT NULL DEFAULT '',
  keywords TEXT NOT NULL DEFAULT '[]',
  created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page INTEGER NOT NULL,
  idx INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scope TEXT NOT NULL,            -- 'all' or a document id
  role TEXT NOT NULL,             -- 'user' | 'assistant'
  content TEXT NOT NULL,
  meta TEXT NOT NULL DEFAULT '{}',-- JSON: sources, status, mode
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_scope ON messages(scope, id);
"""


@contextmanager
def conn():
    path = os.path.abspath(DB_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _lock:
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.executescript(SCHEMA)
        try:
            yield c
            c.commit()
        finally:
            c.close()


def add_document(name, sha, pages, chunks, summary, summary_mode, keywords) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO documents(name, sha256, pages, n_chunks, summary, summary_mode, keywords, created)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (name, sha, pages, len(chunks), summary, summary_mode, json.dumps(keywords), time.time()),
        )
        doc_id = cur.lastrowid
        c.executemany(
            "INSERT INTO chunks(doc_id, page, idx, text) VALUES (?,?,?,?)",
            [(doc_id, ch.page, ch.idx, ch.text) for ch in chunks],
        )
        return doc_id


def find_by_sha(sha):
    with conn() as c:
        r = c.execute("SELECT id FROM documents WHERE sha256=?", (sha,)).fetchone()
        return r["id"] if r else None


def list_documents():
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT id, name, pages, n_chunks, summary_mode, created FROM documents ORDER BY id")]


def get_document(doc_id):
    with conn() as c:
        r = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["keywords"] = json.loads(d["keywords"])
        return d


def delete_document(doc_id) -> bool:
    with conn() as c:
        n = c.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount
        c.execute("DELETE FROM messages WHERE scope=?", (str(doc_id),))
        return n > 0


def chunks_for(doc_id=None):
    """All chunks (optionally one doc), joined with doc name, in stable order."""
    q = "SELECT c.id, c.doc_id, d.name AS doc_name, c.page, c.idx, c.text FROM chunks c JOIN documents d ON d.id=c.doc_id"
    args = ()
    if doc_id is not None:
        q += " WHERE c.doc_id=?"
        args = (doc_id,)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY c.doc_id, c.idx", args)]


def page_text(doc_id, page):
    with conn() as c:
        rows = c.execute("SELECT text FROM chunks WHERE doc_id=? AND page=? ORDER BY idx", (doc_id, page)).fetchall()
        return "\n\n".join(r["text"] for r in rows)


def add_message(scope, role, content, meta=None):
    with conn() as c:
        c.execute("INSERT INTO messages(scope, role, content, meta, created) VALUES (?,?,?,?,?)",
                  (str(scope), role, content, json.dumps(meta or {}), time.time()))


def get_history(scope, limit=100):
    with conn() as c:
        rows = c.execute("SELECT role, content, meta FROM messages WHERE scope=? ORDER BY id DESC LIMIT ?",
                         (str(scope), limit)).fetchall()
    return [{"role": r["role"], "content": r["content"], "meta": json.loads(r["meta"])} for r in reversed(rows)]


def clear_history(scope):
    with conn() as c:
        c.execute("DELETE FROM messages WHERE scope=?", (str(scope),))
