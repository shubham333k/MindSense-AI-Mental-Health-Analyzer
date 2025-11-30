# history_store.py
import sqlite3
from datetime import datetime

DB = "mental_history.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        text TEXT,
        state TEXT,
        confidence REAL,
        rec_ids TEXT,
        feedback TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_event(text, state, confidence, rec_ids):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO events (ts,text,state,confidence,rec_ids) VALUES (?,?,?,?,?)",
              (datetime.utcnow().isoformat(), text, state, confidence, ",".join(rec_ids)))
    conn.commit()
    conn.close()

def record_feedback(event_id, feedback):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("UPDATE events SET feedback=? WHERE id=?", (feedback, event_id))
    conn.commit()
    conn.close()

def fetch_recent(n=50):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id,ts,text,state,confidence,rec_ids,feedback FROM events ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return rows
