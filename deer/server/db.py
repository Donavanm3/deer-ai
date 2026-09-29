"""SQLite storage for users, sessions, datasets, models, training logs and chats."""
import json
import os
import sqlite3
import threading
import time

DATA_DIR = os.path.abspath(os.environ.get("DEER_DATA", os.path.join(os.getcwd(), "deer_data")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL COLLATE NOCASE,
  pw_hash TEXT NOT NULL, salt TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user',
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
  token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
  stats TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
  dataset_id INTEGER REFERENCES datasets(id), config TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'new', error TEXT, step INTEGER NOT NULL DEFAULT 0,
  target_steps INTEGER NOT NULL DEFAULT 0, params INTEGER NOT NULL DEFAULT 0,
  meta TEXT NOT NULL DEFAULT '{}', published INTEGER NOT NULL DEFAULT 0,
  is_default INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS training_log (
  id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
  step INTEGER NOT NULL, loss REAL, metric REAL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_log_model ON training_log(model_id, step);
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS saved (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE, created_at REAL NOT NULL,
  UNIQUE(user_id, message_id));
"""

_local = threading.local()


def db_path():
    return os.path.join(DATA_DIR, "deer.db")


def connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path(), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def conn():
    c = getattr(_local, "conn", None)
    if c is None or getattr(_local, "path", None) != db_path():
        c = connect()
        _local.conn, _local.path = c, db_path()
    return c


def init():
    c = conn()
    c.executescript(SCHEMA)
    c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('allow_registration','1')")
    _migrate(c)
    # anything marked training from a previous run was interrupted
    c.execute("UPDATE models SET status='stopped' WHERE status IN ('training','queued')")


# columns added after the first release; existing databases get them on startup
MIGRATIONS = {
    "models": [("auto_learn", "INTEGER NOT NULL DEFAULT 0"), ("learn_steps", "INTEGER NOT NULL DEFAULT 300"),
               ("learn_min", "INTEGER NOT NULL DEFAULT 10"), ("last_learn_at", "REAL NOT NULL DEFAULT 0")],
    "users": [("test_mode", "TEXT NOT NULL DEFAULT ''"), ("train_ok", "INTEGER NOT NULL DEFAULT 1"),
              ("api_balance", "INTEGER NOT NULL DEFAULT 0")],
}


def _migrate(c):
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, decl in cols:
            if name not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def q(sql, args=()):
    return [dict(r) for r in conn().execute(sql, args).fetchall()]


def q1(sql, args=()):
    r = conn().execute(sql, args).fetchone()
    return dict(r) if r else None


def ex(sql, args=()):
    cur = conn().execute(sql, args)
    return cur.lastrowid


def now():
    return time.time()


def setting(key, default=None):
    r = q1("SELECT value FROM settings WHERE key=?", (key,))
    return r["value"] if r else default


def set_setting(key, value):
    ex("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
       (key, str(value)))


def model_row(mid):
    r = q1("SELECT * FROM models WHERE id=?", (mid,))
    if r:
        r["config"] = json.loads(r["config"])
        r["meta"] = json.loads(r["meta"])
    return r


def model_dir(mid):
    return os.path.join(DATA_DIR, "models", str(int(mid)))


def dataset_dir(did):
    return os.path.join(DATA_DIR, "datasets", str(int(did)))
