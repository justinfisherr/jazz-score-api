"""Vercel function: Spotify search + taste-model scoring.

GET /api/score?q=moanin%20art%20blakey        -> score one track
GET /api/score?title=So%20What&artist=Miles%20Davis
GET /api/playlist?q=jazz                      -> search playlists
GET /api/playlist?id=<playlist_id>            -> score every track in the playlist
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
import genre_bridge  # noqa: E402
from predict import predict_track, predict_tracks  # noqa: E402
from score import (  # noqa: E402
    build_candidate,
    fetch_reccobeats,
    fetch_spotify_audio,
    get_token,
    get_track_by_id,
    search_track,
    search_tracks,
    track_image,
    track_summary,
    track_url,
)

SPOTIFY_API = "https://api.spotify.com/v1"
RECCOBEATS_URL = "https://api.reccobeats.com/v1/audio-features"


def search_playlists(token, q):
    resp = requests.get(
        f"{SPOTIFY_API}/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": q, "type": "playlist", "limit": 6},
        timeout=30,
    )
    resp.raise_for_status()
    return [p for p in resp.json().get("playlists", {}).get("items", []) if p]


def search_albums(token, q):
    resp = requests.get(
        f"{SPOTIFY_API}/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": q, "type": "album", "limit": 10},
        timeout=30,
    )
    resp.raise_for_status()
    return [a for a in resp.json().get("albums", {}).get("items", []) if a]


def album_summary(album):
    artists = album.get("artists") or []
    year = None
    if album.get("release_date"):
        try:
            year = int(album["release_date"][:4])
        except ValueError:
            year = None
    return {
        "id": album["id"],
        "name": album["name"],
        "artist": ", ".join(a["name"] for a in artists),
        "image": (album.get("images") or [{}])[0].get("url") if album.get("images") else None,
        "year": year,
        "track_count": album.get("total_tracks"),
        "url": (album.get("external_urls") or {}).get("spotify"),
    }


def get_album_tracks(token, album_id):
    items = []
    url = f"{SPOTIFY_API}/albums/{album_id}/tracks"
    params = {"limit": 50}
    while url and len(items) < 100:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get("items", []))
        url = data.get("next")
        params = {}
    return items


def get_playlist_tracks(token, pid):
    items = []
    url = f"{SPOTIFY_API}/playlists/{pid}/tracks"
    params = {
        "limit": 50,
        "fields": "items(track(id,name,artists,album(name,images),duration_ms,popularity,external_urls)),next",
    }
    while url and len(items) < 100:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items.extend(i["track"] for i in data.get("items", []) if i.get("track"))
        url = data.get("next")
        params = {}
    return items


def fetch_reccobeats_batch(ids):
    resp = requests.get(RECCOBEATS_URL, params={"ids": ",".join(ids)}, timeout=30)
    if resp.status_code != 200:
        return {}
    out = {}
    for item in resp.json().get("content", []):
        href = item.get("href", "")
        sid = href.rstrip("/").split("/")[-1] if href else None
        if sid:
            out[sid] = {
                "acousticness": item.get("acousticness"),
                "danceability": item.get("danceability"),
                "spotify_energy": item.get("energy"),
                "instrumentalness": item.get("instrumentalness"),
                "liveness": item.get("liveness"),
                "loudness": item.get("loudness"),
                "speechiness": item.get("speechiness"),
                "spotify_valence": item.get("valence"),
            }
    return out


def fetch_spotify_audio_batch(token, ids):
    if not ids:
        return {}
    resp = requests.get(
        f"{SPOTIFY_API}/audio-features",
        headers={"Authorization": f"Bearer {token}"},
        params={"ids": ",".join(ids)},
        timeout=30,
    )
    if resp.status_code != 200:
        return {}
    return {a["id"]: a for a in resp.json().get("audio_features", []) if a}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        params = parse_qs(urlparse(self.path).query)
        try:
            if path.startswith("/api/search"):
                self._search_results(params)
            elif path.startswith("/api/album"):
                self._album(params)
            elif path.startswith("/api/playlist"):
                self._playlist(params)
            elif path.startswith("/api/genres"):
                self._json(200, {"genres": genre_bridge.genres()})
            else:
                self._score(params)
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)})

    def _search_results(self, params):
        q = (params.get("q") or [""])[0].strip()
        if not q:
            self._json(400, {"error": "provide q"})
            return
        token = get_token()
        tracks = search_tracks(token, q, limit=10)
        keys = db.reviewed_keys()
        results = []
        for t in tracks:
            summary = track_summary(t)
            summary["reviewed"] = db.is_reviewed(summary["title"], summary["artist"], keys)
            results.append(summary)
        self._json(200, {"results": results})

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/rate"):
            self._rate()
        else:
            self._json(404, {"error": "not found"})

    def _rate(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            track = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        title = (track.get("title") or "").strip()
        artist = (track.get("artist") or "").strip()
        rating = track.get("rating")
        if not title or not artist or rating is None:
            self._json(400, {"error": "title, artist, and rating are required"})
            return
        existing = db.get_track(title, artist) or {}
        merged = {**existing, **{k: v for k, v in track.items() if v is not None}}
        merged["title"] = title
        merged["artist"] = artist
        merged["rating"] = rating
        db.upsert_track(merged)
        self._json(200, {"ok": True, "title": title, "artist": artist, "rating": rating})

    def _score(self, params):
        q = (params.get("q") or [""])[0].strip()
        sid = (params.get("id") or [""])[0].strip()
        title = (params.get("title") or [""])[0].strip()
        artist = (params.get("artist") or [""])[0].strip()
        genre = (params.get("genre") or [""])[0].strip()
        if not q and not sid and not (title and artist):
            self._json(400, {"error": "provide q, id, or title+artist"})
            return
        token = get_token()
        track = get_track_by_id(token, sid) if sid else search_track(token, q, title or None, artist or None)
        if not track:
            self._json(404, {"error": "no track found"})
            return
        sfa = fetch_spotify_audio(token, track["id"])
        recco = fetch_reccobeats(track["id"])
        candidate = genre_bridge.apply(build_candidate(track, sfa, recco), genre)
        score = predict_track(candidate)
        self._json(200, {
            "title": candidate["title"],
            "artist": candidate["artist"],
            "year": candidate["year"],
            "score": score,
            "tempo": candidate["tempo"],
            "popularity": candidate["audio_features"].get("popularity"),
            "image": track_image(track),
            "url": track_url(track),
            "spotify_id": track["id"],
            "audio_only": True,
            "genre": genre or None,
        })

    def _playlist(self, params):
        q = (params.get("q") or [""])[0].strip()
        pid = (params.get("id") or [""])[0].strip()
        token = get_token()

        if q and not pid:
            playlists = search_playlists(token, q)
            self._json(200, {"playlists": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "owner": (p.get("owner") or {}).get("display_name"),
                    "image": (p.get("images") or [{}])[0].get("url") if p.get("images") else None,
                    "track_count": (p.get("tracks") or {}).get("total"),
                    "url": (p.get("external_urls") or {}).get("spotify"),
                }
                for p in playlists
            ]})
            return

        if not pid:
            self._json(400, {"error": "provide q (search) or id (playlist id)"})
            return

        meta_resp = requests.get(
            f"{SPOTIFY_API}/playlists/{pid}",
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "name,owner(display_name),images,tracks(total),external_urls"},
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        tracks = get_playlist_tracks(token, pid)
        ids = [t["id"] for t in tracks]
        recco = fetch_reccobeats_batch(ids)
        sfa = fetch_spotify_audio_batch(token, ids)

        candidates = [build_candidate(t, sfa.get(t["id"], {}), recco.get(t["id"])) for t in tracks]
        scores = predict_tracks(candidates)

        self._json(200, {
            "name": meta.get("name"),
            "owner": (meta.get("owner") or {}).get("display_name"),
            "image": (meta.get("images") or [{}])[0].get("url") if meta.get("images") else None,
            "url": (meta.get("external_urls") or {}).get("spotify"),
            "tracks": [
                {
                    "title": c["title"],
                    "artist": c["artist"],
                    "score": s,
                    "image": track_image(t),
                    "url": track_url(t),
                }
                for t, c, s in zip(tracks, candidates, scores)
            ],
        })

    def _album(self, params):
        q = (params.get("q") or [""])[0].strip()
        aid = (params.get("id") or [""])[0].strip()
        token = get_token()

        if q and not aid:
            albums = search_albums(token, q)
            keys = db.reviewed_keys()
            out = []
            for a in albums:
                summary = album_summary(a)
                summary["reviewed"] = db.is_reviewed(summary["name"], summary["artist"], keys)
                out.append(summary)
            self._json(200, {"albums": out})
            return

        if not aid:
            self._json(400, {"error": "provide q (search) or id (album id)"})
            return

        meta_resp = requests.get(
            f"{SPOTIFY_API}/albums/{aid}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        album_artist = ", ".join(a["name"] for a in meta.get("artists", []))
        album_image = (meta.get("images") or [{}])[0].get("url") if meta.get("images") else None
        year = None
        if meta.get("release_date"):
            try:
                year = int(meta["release_date"][:4])
            except ValueError:
                year = None

        tracks = get_album_tracks(token, aid)
        ids = [t["id"] for t in tracks]
        recco = fetch_reccobeats_batch(ids)
        sfa = fetch_spotify_audio_batch(token, ids)

        candidates = [build_candidate(t, sfa.get(t["id"], {}), recco.get(t["id"])) for t in tracks]
        scores = predict_tracks(candidates)

        self._json(200, {
            "name": meta.get("name"),
            "artist": album_artist,
            "year": year,
            "image": album_image,
            "url": (meta.get("external_urls") or {}).get("spotify"),
            "album_id": aid,
            "reviewed": db.is_reviewed(meta.get("name"), album_artist),
            "tracks": [
                {
                    "title": c["title"],
                    "artist": c["artist"],
                    "score": s,
                    "image": album_image,
                    "url": track_url(t),
                    "spotify_id": t["id"],
                }
                for t, c, s in zip(tracks, candidates, scores)
            ],
        })

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
