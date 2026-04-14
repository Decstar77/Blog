import sqlite3
import os
from contextlib import contextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

DB_PATH = os.environ.get("DB_PATH", "/data/tracker.db")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS page_visits (
                page  TEXT PRIMARY KEY,
                visits INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


init_db()


@app.post("/ping")
def ping(page: str = Query(..., description="Page identifier, e.g. 'index' or 'articles'")):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO page_visits (page, visits) VALUES (?, 1)
            ON CONFLICT(page) DO UPDATE SET visits = visits + 1
        """, (page,))
        conn.commit()
        row = conn.execute("SELECT visits FROM page_visits WHERE page = ?", (page,)).fetchone()
    return {"page": page, "visits": row["visits"]}


@app.get("/stats")
def stats():
    with get_db() as conn:
        rows = conn.execute("SELECT page, visits FROM page_visits ORDER BY visits DESC").fetchall()
    return {"pages": [{"page": r["page"], "visits": r["visits"]} for r in rows]}


@app.get("/health")
def health():
    return {"status": "ok"}
