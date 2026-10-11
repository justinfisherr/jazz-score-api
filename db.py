"""Postgres access for the jazz taste dataset.

The `tracks` table is the live source of truth. Each row stores a full track
dict in a JSONB `data` column, with title/artist/spotify_id/rating denormalized
for deduping and quick reads. Reads/writes go through DATABASE_URL (or
POSTGRES_URL) from the environment.
"""

import json
import os

import psycopg2
from psycopg2.extras import Json

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    spotify_id TEXT,
    rating NUMERIC,
    data JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (title, artist)
);
"""


def get_url():
    return (
        os.environ.get("DATABASE_URL")
        or os.environ.get("POSTGRES_URL")
        or os.environ.get("POSTGRES_URL_NON_POOLING")
    )


def get_conn():
    url = get_url()
    if not url:
        raise RuntimeError("No DATABASE_URL / POSTGRES_URL set")
    return psycopg2.connect(url)


def init_schema():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(_SCHEMA)


def fetch_tracks():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT data FROM tracks ORDER BY id")
        return [r[0] for r in cur.fetchall()]


def get_track(title, artist):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT data FROM tracks WHERE title = %s AND artist = %s", (title, artist))
        row = cur.fetchone()
        return row[0] if row else None


def upsert_track(track):
    title = (track.get("title") or "").strip()
    artist = (track.get("artist") or "").strip()
    if not title or not artist:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tracks (title, artist, spotify_id, rating, data)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (title, artist) DO UPDATE
            SET spotify_id = EXCLUDED.spotify_id,
                rating = EXCLUDED.rating,
                data = EXCLUDED.data,
                updated_at = now()
            """,
            (title, artist, track.get("spotify_id"), track.get("rating"), Json(track)),
        )


def bulk_upsert(tracks):
    """Upsert many tracks over a single connection (fast migration path)."""
    rows = []
    for t in tracks:
        title = (t.get("title") or "").strip()
        artist = (t.get("artist") or "").strip()
        if title and artist:
            rows.append((title, artist, t.get("spotify_id"), t.get("rating"), Json(t)))
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO tracks (title, artist, spotify_id, rating, data)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (title, artist) DO UPDATE
            SET spotify_id = EXCLUDED.spotify_id,
                rating = EXCLUDED.rating,
                data = EXCLUDED.data,
                updated_at = now()
            """,
            rows,
        )
    return len(rows)
