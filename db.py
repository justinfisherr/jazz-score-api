"""Postgres access for the jazz taste dataset.

The `tracks` table is the live source of truth. Each row stores a full track
dict in a JSONB `data` column, with title/artist/spotify_id/rating denormalized
for deduping and quick reads. Reads/writes go through DATABASE_URL (or
POSTGRES_URL) from the environment.
"""

import json
import os
import re

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

CREATE TABLE IF NOT EXISTS pending_reviews (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
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


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _norm_title(s):
    s = _norm(s)
    s = re.sub(r"\s*[-–]\s*(19|20)\d\d\s+.*$", "", s)
    s = re.sub(r"\s*[\(\[].*?(?:remaster|mono|take \d|alternate|live|version).*?[\)\]]", "", s)
    return s.strip()


def reviewed_keys():
    """Return normalized (title, artist) tuples for every reviewed track/album."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT title, artist FROM tracks")
        return [(_norm_title(r[0]), _norm(r[1])) for r in cur.fetchall()]


def is_reviewed(title, artist, keys=None):
    """True if the dataset already contains this (title, artist) — fuzzy on
    version suffixes and artist strings so 'Giant Steps - 2020 Remaster' matches
    an already-reviewed 'Giant Steps'."""
    if keys is None:
        keys = reviewed_keys()
    tn = _norm_title(title)
    an = _norm(artist)
    for rt, ra in keys:
        if rt != tn:
            continue
        if ra == an or (ra and an and (ra in an or an in ra)):
            return True
    return False


def match_review(title, artist, tracks=None):
    """Return the full dataset track matching (title, artist) via fuzzy matching,
    or None. Pass `tracks` to avoid re-fetching the table per track."""
    if tracks is None:
        tracks = fetch_tracks()
    tn = _norm_title(title)
    an = _norm(artist)
    for t in tracks:
        if _norm_title(t.get("title")) == tn:
            ta = _norm(t.get("artist") or "")
            if ta == an or (ta and an and (ta in an or an in ta)):
                return t
    return None


def add_pending_review(review):
    title = (review.get("title") or "").strip()
    artist = (review.get("artist") or "").strip()
    if not title or not artist:
        return None
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pending_reviews (title, artist, data) VALUES (%s, %s, %s) RETURNING id",
            (title, artist, Json(review)),
        )
        return cur.fetchone()[0]


def get_pending_reviews():
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, data FROM pending_reviews ORDER BY id")
        return [{"id": r[0], **r[1]} for r in cur.fetchall()]


def clear_pending_reviews(ids=None):
    with get_conn() as conn, conn.cursor() as cur:
        if ids:
            cur.execute("DELETE FROM pending_reviews WHERE id = ANY(%s)", (list(ids),))
        else:
            cur.execute("DELETE FROM pending_reviews")
        return cur.rowcount
