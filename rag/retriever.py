import sqlite3, numpy as np
from dataclasses import dataclass
from typing import List

DB_PATH = "rag_store.db"

@dataclass
class RetrievedChunk:
    chunk_id: str
    source_file: str
    content: str
    score: float            # similarity score (0–1, higher = more relevant)

def init_db(db_path=DB_PATH):
    """Create the table if it doesn't exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id        TEXT PRIMARY KEY,
                source    TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB NOT NULL   -- float32 bytes
            )
        """)

def clear_db(db_path=DB_PATH):
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM chunks")

def insert_chunks(ids, sources, contents, embeddings, db_path=DB_PATH):
    rows = [
        (cid, src, text, emb.astype(np.float32).tobytes())
        for cid, src, text, emb in zip(ids, sources, contents, embeddings)
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)", rows
        )

def retrieve(query_emb: np.ndarray, top_k=4, db_path=DB_PATH):
    # fetch everything (works fine for small data)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, source, content, embedding FROM chunks"
        ).fetchall()

    if not rows:
        return []

    ids, sources, contents, blobs = zip(*rows)
    dim = query_emb.shape[0]
    # convert all stored embeddings back into matrix
    matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(rows), dim)

    # Dot product = cosine similarity (because vectors are L2-normalised)
    scores = matrix @ query_emb

    # Get indices of top-k highest scores
    top_idx = np.argsort(scores)[-top_k:][::-1]

    return [
        RetrievedChunk(ids[i], sources[i], contents[i], float(scores[i]))
        for i in top_idx
    ]

def count_chunks(db_path=DB_PATH) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]