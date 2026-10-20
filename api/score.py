"""Vercel function: Spotify search + taste-model scoring.

GET /api/score?q=moanin%20art%20blakey        -> score one track
GET /api/score?title=So%20What&artist=Miles%20Davis
GET /api/playlist?q=jazz                      -> search playlists
GET /api/playlist?id=<playlist_id>            -> score every track in the playlist
"""

import json
import re
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
import enrich  # noqa: E402
import genre_bridge  # noqa: E402

db.init_schema()
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
APP_PASSWORD = os.environ.get("APP_PASSWORD", "4416")
GH_TOKEN = os.environ.get("GH_WORKFLOW_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "justinfisherr/claude-metrics")
GH_WORKFLOW = os.environ.get("GH_WORKFLOW", "train.yml")
GH_API = "https://api.github.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
PLAYBACK_TOKEN_URL = os.environ.get("PLAYBACK_TOKEN_URL", "https://transcribe-xi-virid.vercel.app/api/spotify-token")
_playback_token_cache = {"token": None, "expires_at": 0}
JAZZ_EXTRA = {"stretch"}


def get_user_token():
    now = time.time()
    if _playback_token_cache.get("token") and _playback_token_cache.get("expires_at", 0) > now + 60:
        return _playback_token_cache["token"]
    resp = requests.get(PLAYBACK_TOKEN_URL, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"token broker returned {resp.status_code}")
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("token broker returned no token")
    expires = int(data.get("expires_in") or 3600)
    _playback_token_cache["token"] = token
    _playback_token_cache["expires_at"] = now + expires
    return token


def is_jazz_playlist(name):
    n = (name or "").strip().lower()
    return "jazz" in n or "bop" in n or n in JAZZ_EXTRA or "liked" in n or n == "likes"


def playlist_summary(p):
    return {
        "id": p["id"],
        "name": p["name"],
        "owner": (p.get("owner") or {}).get("display_name"),
        "image": (p.get("images") or [{}])[0].get("url") if p.get("images") else None,
        "track_count": (p.get("tracks") or {}).get("total"),
        "public": p.get("public"),
        "url": (p.get("external_urls") or {}).get("spotify"),
        "jazz": is_jazz_playlist(p.get("name")),
    }


def fetch_owned_playlists(token):
    headers = {"Authorization": f"Bearer {token}"}
    me = requests.get(f"{SPOTIFY_API}/me", headers=headers, timeout=20).json()
    mine = me.get("id")
    out = []
    url = f"{SPOTIFY_API}/me/playlists"
    params = {"limit": 50}
    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for p in data.get("items", []):
            if p and (p.get("owner") or {}).get("id") == mine:
                out.append(playlist_summary(p))
        url = data.get("next")
        params = {}
    return out


def analyze_playlist(p, token, all_tracks):
    tracks = get_playlist_tracks(token, p["id"])[:100]
    rows = []
    matched = []
    for t in tracks:
        artist = ", ".join(a.get("name", "") for a in t.get("artists", []))
        existing = db.match_review(t.get("name"), artist, all_tracks)
        if existing:
            matched.append(existing)
        rows.append({
            "title": t.get("name"),
            "artist": artist,
            "matched": bool(existing),
            "actual": existing.get("rating") if existing else None,
            "predicted": None,
        })
    ratings = [m.get("rating") for m in matched if m.get("rating") is not None]
    replay = [m.get("replayability") for m in matched if m.get("replayability") is not None]
    energies = [m.get("energy") for m in matched if m.get("energy") is not None]
    replay_avg = (sum(replay) / len(replay)) if replay else None
    if replay_avg is not None and replay_avg > 1:
        replay_avg = replay_avg / 10
    top = lambda field, key: [v for v, _ in Counter(x for m in matched for x in (m.get(field) or [])).most_common(3)] if field in ("moods", "subgenres") else [v for v, _ in Counter(m.get(field) for m in matched if m.get(field)).most_common(3)]
    return {
        **p,
        "track_count": len(tracks),
        "matched_count": len(matched),
        "avg_actual": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "avg_predicted": None,
        "avg_energy": round(sum(energies) / len(energies), 1) if energies else None,
        "avg_replayability": round(replay_avg, 3) if replay_avg is not None else None,
        "top_moods": top("moods", "mood"),
        "top_eras": top("era", "era"),
        "tracks": rows,
    }


def gh_headers():
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def latest_train_run():
    resp = requests.get(
        f"{GH_API}/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/runs",
        headers=gh_headers(),
        params={"per_page": 1},
        timeout=20,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    runs = resp.json().get("workflow_runs", [])
    return runs[0] if runs else None


def run_summary(run):
    if not run:
        return {"status": "idle", "conclusion": None, "url": None}
    return {
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "url": run.get("html_url"),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "run_number": run.get("run_number"),
        "display_title": run.get("display_title"),
    }


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


def build_taste_profile(tracks):
    rated = [t for t in tracks if t.get("entity_type") != "album" and t.get("rating") is not None]
    top = sorted(rated, key=lambda t: t.get("rating", 0), reverse=True)[:14]
    low = sorted(rated, key=lambda t: t.get("rating", 0))[:8]
    recent = sorted(rated, key=lambda t: (t.get("date_added") or ""), reverse=True)[:45]

    def mode(field):
        return Counter(v for t in rated for v in ([t.get(field)] if isinstance(t.get(field), str) else (t.get(field) or []))).most_common(8)

    artists = {}
    for t in rated:
        a = t.get("artist")
        if not a:
            continue
        stats = artists.setdefault(a, {"n": 0, "sum": 0.0})
        stats["n"] += 1
        stats["sum"] += t.get("rating", 0)
    top_artists = [
        {"artist": a, "avg": round(v["sum"] / v["n"], 2), "n": v["n"]}
        for a, v in artists.items() if v["n"] >= 2
    ]
    top_artists.sort(key=lambda x: (-x["avg"], -x["n"]))
    low_artists = [a for a in top_artists if a["avg"] <= 4.5]
    album_stats = {}
    for t in rated:
        if not t.get("album"):
            continue
        key = (t.get("album"), t.get("artist"))
        stats = album_stats.setdefault(key, {"n": 0, "sum": 0.0})
        stats["n"] += 1
        stats["sum"] += t.get("rating", 0)
    top_albums = [
        {"album": k[0], "artist": k[1], "avg": round(v["sum"] / v["n"], 2), "n": v["n"]}
        for k, v in album_stats.items()
    ]
    top_albums.sort(key=lambda x: (-x["avg"], -x["n"]))
    return {
        "top_rated": [{"title": t.get("title"), "artist": t.get("artist"), "rating": t.get("rating")} for t in top],
        "misses": [{"title": t.get("title"), "artist": t.get("artist"), "rating": t.get("rating")} for t in low],
        "avoid_recent": [{"title": t.get("title"), "artist": t.get("artist")} for t in recent],
        "top_artists": top_artists[:12],
        "low_artists": low_artists[:8],
        "top_albums": top_albums[:10],
        "top_eras": [{"era": k, "n": n} for k, n in mode("era")],
        "top_subgenres": [{"subgenre": k, "n": n} for k, n in mode("subgenres")],
        "top_moods": [{"mood": k, "n": n} for k, n in mode("moods")],
        "top_instruments": [{"instrument": k, "n": n} for k, n in mode("primary_instrument")],
    }


def ask_openrouter(kind, prompt, count, profile, reject=None):
    shape = '{"recommendations":[{"title":"...","artist":"...","reason":"...","key_players":["Name - instrument"],"confidence":"high|medium|low","spotify_query":"..."}]}'
    kind_rule = {
        "track": "Every recommendation must be ONE TRACK. title is the track title, artist is the performing artist. Never return an album or playlist.",
        "album": "Every recommendation must be ONE ALBUM. title is the album title, artist is the album artist. Never return a single track.",
        "playlist": "Every recommendation must be ONE PLAYLIST. title is a short playlist name, artist is the curator or vibe, and spotify_query must be a Spotify playlist search phrase.",
    }[kind]
    system = (
        "You are Justin's jazz concierge. You know his taste model and his library. "
        "Recommend jazz recordings only unless the request explicitly asks for another genre. "
        f"Return exactly {count + 3} distinct recommendations. "
        "Recommend music he has not already rated. Be specific and lean into gaps, but balance about "
        "70% fit to his highest-rated eras, instruments, artists, collaborators, and albums with 30% exploration. "
        "Favor artists one or two degrees away from his top artists (same scene, sidemen, or label) and avoid low_artists. "
        "At least half of the picks should be by top_artists or by musicians who played on their records. "
        "Avoid the most canonical names (Coltrane, Miles Davis, Wayne Shorter, Herbie Hancock) unless the request names them. "
        "A pick that clearly sits outside his taste is a failed recommendation. "
        f"{kind_rule} "
        "For every track recommendation, include the actual performing band in key_players, "
        "using strings like 'Name - instrument' (leader plus sidemen). Do not invent personnel: "
        "return [] if you are not sure. Mention the band in the reason when it helps. "
        "spotify_query must be the exact search phrase that will find that item on Spotify. "
        f"Return JSON only, exactly this shape: {shape}"
    )
    user = json.dumps({
        "kind": kind,
        "request": prompt or "Recommend something that fits my taste right now.",
        "count": count,
        "taste_profile": profile,
        "do_not_return": reject or [],
        "rules": [
            "Never recommend a track/album already represented in avoid_recent, top_rated, or do_not_return.",
            "Use known real recordings only; do not invent titles or personnel.",
            "The reason should be one or two specific sentences, not hype.",
        ],
    }, ensure_ascii=False)
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://claude-metrics-five.vercel.app",
            "X-Title": "Jazz Taste Analysis",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.7,
            "max_tokens": 6000,
            "reasoning": {"enabled": False},
        },
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    content = resp.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("model returned an empty response")
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
    try:
        data = json.loads(content)
        recs = data.get("recommendations") or []
    except json.JSONDecodeError:
        # Reasoning models occasionally truncate the final object. Recover every
        # complete object we can and drop the unfinished tail.
        recs = []
        depth = 0
        start = None
        in_string = False
        escaped = False
        for i, ch in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth:
                    depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(content[start:i + 1])
                        if isinstance(obj, dict) and obj.get("title"):
                            recs.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
        if not recs:
            raise RuntimeError("model did not return valid JSON")
    return [r for r in recs if isinstance(r, dict) and r.get("title")]


def resolve_track_recommendation(rec, token, all_tracks=None):
    title = str(rec.get("title") or "").strip()
    artist = str(rec.get("artist") or "").strip()
    query = str(rec.get("spotify_query") or f"{title} {artist}").strip()
    track = search_track(token, query, title or None, artist or None)
    if not track:
        return None
    sfa = fetch_spotify_audio(token, track["id"])
    recco = fetch_reccobeats(track["id"])
    candidate = build_candidate(track, sfa, recco)
    score = predict_track(candidate)
    all_tracks = all_tracks if all_tracks is not None else db.fetch_tracks()
    existing = db.match_review(candidate["title"], candidate["artist"], all_tracks)
    if not existing:
        existing = next((t for t in all_tracks if t.get("spotify_id") and t.get("spotify_id") == track["id"]), None)
    existing = existing or {}
    return {
        "kind": "track",
        "title": candidate["title"],
        "artist": candidate["artist"],
        "album": (track.get("album") or {}).get("name"),
        "year": candidate["year"],
        "image": track_image(track),
        "url": track_url(track),
        "spotify_id": track["id"],
        "score": score,
        "reason": rec.get("reason"),
        "key_players": (existing.get("key_players") or []) if existing.get("key_players") else [
            p for p in (rec.get("key_players") or []) if isinstance(p, str) and " - " in p
        ][:20],
        "confidence": rec.get("confidence"),
        "reviewed": bool(existing),
        "rating": existing.get("rating"),
        "notes": existing.get("notes"),
        "liked": bool(existing.get("liked")),
    }


def resolve_album_recommendation(rec, token):
    title = str(rec.get("title") or "").strip()
    artist = str(rec.get("artist") or "").strip()
    query = str(rec.get("spotify_query") or f"{title} {artist}").strip()
    albums = search_albums(token, query)
    if not albums:
        return None
    album = albums[0]
    summary = album_summary(album)
    summary.update({
        "kind": "album",
        "reason": rec.get("reason"),
        "confidence": rec.get("confidence"),
        "reviewed": db.is_reviewed(summary["name"], summary["artist"]),
    })
    return summary


def resolve_playlist_recommendation(rec, token):
    query = str(rec.get("spotify_query") or rec.get("title") or "").strip()
    playlists = search_playlists(token, query)
    if not playlists:
        return None
    p = playlists[0]
    return {
        "kind": "playlist",
        "title": p.get("name"),
        "artist": (p.get("owner") or {}).get("display_name"),
        "image": (p.get("images") or [{}])[0].get("url") if p.get("images") else None,
        "track_count": (p.get("tracks") or {}).get("total"),
        "url": (p.get("external_urls") or {}).get("spotify"),
        "spotify_id": p.get("id"),
        "reason": rec.get("reason"),
        "confidence": rec.get("confidence"),
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        params = parse_qs(urlparse(self.path).query)
        try:
            if path.startswith("/api/search"):
                self._search_results(params)
            elif path.startswith("/api/album"):
                self._album(params)
            elif path.startswith("/api/playlists"):
                self._playlists(params)
            elif path.startswith("/api/playlist"):
                self._playlist(params)
            elif path.startswith("/api/pending"):
                self._json(200, {"pending": db.get_pending_reviews()})
            elif path.startswith("/api/genres"):
                self._json(200, {"genres": genre_bridge.genres()})
            elif path.startswith("/api/train"):
                self._train_status()
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
        all_tracks = db.fetch_tracks()
        results = []
        for t in tracks:
            summary = track_summary(t)
            existing = db.match_review(summary["title"], summary["artist"], all_tracks) or {}
            summary["reviewed"] = bool(existing)
            summary["rating"] = existing.get("rating")
            summary["notes"] = existing.get("notes")
            results.append(summary)
        self._json(200, {"results": results})

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-app-password")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/auth"):
            self._auth_check()
            return
        if self.headers.get("x-app-password") != APP_PASSWORD:
            self._json(401, {"error": "unauthorized"})
            return
        if path.startswith("/api/spotify-playback-token"):
            self._playback_token()
        elif path.startswith("/api/playlist/add"):
            self._add_to_playlist()
        elif path.startswith("/api/recommend"):
            self._recommend()
        elif path.startswith("/api/enrich"):
            self._enrich()
        elif path.startswith("/api/train"):
            self._trigger_train()
        elif path.startswith("/api/rate/delete"):
            self._unrate()
        elif path.startswith("/api/pending/clear"):
            self._clear_pending()
        elif path.startswith("/api/pending"):
            self._add_pending()
        elif path.startswith("/api/rate"):
            self._rate()
        else:
            self._json(404, {"error": "not found"})

    def _auth_check(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        if body.get("password") == APP_PASSWORD:
            self._json(200, {"ok": True})
        else:
            self._json(401, {"error": "wrong password"})

    def _playback_token(self):
        try:
            token = get_user_token()
            self._json(200, {"access_token": token, "expires_in": int(max(0, _playback_token_cache["expires_at"] - time.time()))})
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"playback token unavailable: {exc}"})

    def _playlists(self, params):
        try:
            token = get_user_token()
            playlists = [p for p in fetch_owned_playlists(token) if p["jazz"]]
            try:
                saved = requests.get(
                    f"{SPOTIFY_API}/me/tracks",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"limit": 1},
                    timeout=20,
                ).json()
                liked_count = saved.get("total")
            except Exception:  # noqa: BLE001
                liked_count = None
            playlists = [{
                "id": "__liked__",
                "name": "Liked Songs",
                "owner": "You",
                "image": None,
                "track_count": liked_count,
                "public": False,
                "url": "https://open.spotify.com/collection/tracks",
                "jazz": True,
                "special": "liked",
            }] + playlists
            if (params.get("analyze") or ["0"])[0] == "1":
                all_tracks = db.fetch_tracks()
                real = [p for p in playlists if p.get("special") != "liked"]
                with ThreadPoolExecutor(max_workers=6) as pool:
                    playlists = list(pool.map(lambda p: analyze_playlist(p, token, all_tracks), real))
            self._json(200, {"playlists": playlists})
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"playlists unavailable: {exc}"})

    def _add_to_playlist(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        playlist_id = (body.get("playlist_id") or "").strip()
        spotify_id = (body.get("spotify_id") or "").strip()
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        if not playlist_id or not spotify_id or not title or not artist:
            self._json(400, {"error": "playlist_id, spotify_id, title and artist are required"})
            return
        try:
            token = get_user_token()
            if playlist_id == "__liked__":
                resp = requests.put(
                    f"{SPOTIFY_API}/me/tracks",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"ids": [spotify_id]},
                    timeout=30,
                )
                if resp.status_code not in (200, 201, 204):
                    raise RuntimeError(f"spotify like failed ({resp.status_code}): {resp.text[:200]}")
            else:
                resp = requests.post(
                    f"{SPOTIFY_API}/playlists/{playlist_id}/tracks",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"uris": [f"spotify:track:{spotify_id}"]},
                    timeout=30,
                )
                if resp.status_code not in (200, 201):
                    raise RuntimeError(f"spotify add failed ({resp.status_code}): {resp.text[:200]}")
            existing = db.get_track(title, artist) or {}
            merged = {
                **existing,
                "title": title,
                "artist": artist,
                "spotify_id": existing.get("spotify_id") or spotify_id,
                "album": body.get("album") or existing.get("album"),
                "year": body.get("year") or existing.get("year"),
                "entity_type": existing.get("entity_type") or "track",
                "liked": True,
                "liked_at": datetime.utcnow().isoformat(),
                "discovered_from": existing.get("discovered_from") or "ai-like",
                "key_players": body.get("key_players") or existing.get("key_players") or [],
            }
            if not merged.get("date_added"):
                merged["date_added"] = datetime.utcnow().date().isoformat()
            try:
                merged = enrich.enrich_track(merged, db.fetch_tracks(), token)
            except Exception:  # noqa: BLE001
                pass
            merged["liked"] = True
            db.upsert_track(merged)
            self._json(200, {"ok": True, "liked": True, "playlist_id": playlist_id})
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"could not add to playlist: {exc}"})

    def _recommend(self):
        if not OPENROUTER_API_KEY:
            self._json(503, {"error": "AI recommendations are not configured (OPENROUTER_API_KEY missing)"})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        kind = body.get("kind") if body.get("kind") in ("track", "album", "playlist") else "track"
        prompt = str(body.get("prompt") or "").strip()
        count = max(1, min(10, int(body.get("count") or (10 if kind == "track" else 2))))
        tracks = db.fetch_tracks()
        profile = build_taste_profile(tracks)
        initial_reject = [str(x)[:140] for x in (body.get("exclude") or [])][:30]
        try:
            recs = ask_openrouter(kind, prompt, min(count, 6) if kind == "track" else count + 3, profile, reject=initial_reject)
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"openrouter: {exc}"})
            return
        token = get_token()
        existing_tracks = db.fetch_tracks()
        out = []
        seen = set()
        rejected = list(initial_reject)

        def consume(candidates):
            for rec in candidates:
                if len(out) >= count + 4:
                    return
                try:
                    if kind == "track":
                        item = resolve_track_recommendation(rec, token, existing_tracks)
                        if item and not item.get("reviewed"):
                            key = (item["title"].lower(), item["artist"].lower())
                            if key not in seen:
                                seen.add(key)
                                out.append(item)
                        else:
                            rejected.append(f"{rec.get('title')} — {rec.get('artist')}")
                    elif kind == "album":
                        item = resolve_album_recommendation(rec, token)
                        if item and not item.get("reviewed"):
                            key = (item["name"].lower(), item["artist"].lower())
                            if key not in seen:
                                seen.add(key)
                                out.append(item)
                        else:
                            rejected.append(f"{rec.get('title')} — {rec.get('artist')}")
                    else:
                        item = resolve_playlist_recommendation(rec, token)
                        if item:
                            key = (item["title"] or "").lower()
                            if key not in seen:
                                seen.add(key)
                                out.append(item)
                        else:
                            rejected.append(f"{rec.get('title')} — {rec.get('artist')}")
                except Exception:  # noqa: BLE001
                    continue

        consume(recs)
        if kind == "track":
            attempts = 0
            while len(out) < count and attempts < 2:
                attempts += 1
                reject = rejected[:] + [f"{r.get('title')} — {r.get('artist')}" for r in out]
                try:
                    more = ask_openrouter("track", prompt, 6, profile, reject=reject[-25:])
                    before = len(out)
                    consume(more)
                    if len(out) == before:
                        break
                except Exception:  # noqa: BLE001
                    break
        if kind == "track":
            # Let the taste model rank the AI's candidates instead of trusting
            # the first three the language model happened to name.
            out.sort(key=lambda r: r.get("score") or 0, reverse=True)
        out = out[:count]
        self._json(200, {
            "kind": kind,
            "prompt": prompt,
            "model": OPENROUTER_MODEL,
            "recommendations": out,
            "message": None if out else "The model only returned things you have already rated. Try a more specific request.",
            "profile": {
                "tracks": len([t for t in tracks if t.get("entity_type") != "album"]),
                "top_eras": profile["top_eras"][:4],
                "top_artists": profile["top_artists"][:6],
            },
        })

    def _enrich(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        if not title or not artist:
            self._json(400, {"error": "title and artist are required"})
            return
        existing = db.get_track(title, artist) or {}
        track = {**existing, **{k: v for k, v in body.items() if v is not None}}
        track["title"], track["artist"] = title, artist
        try:
            token = get_token()
            enriched = enrich.enrich_track(track, db.fetch_tracks(), token, use_agent=bool(body.get("use_agent")))
        except Exception as exc:  # noqa: BLE001
            enriched = {**track, "needs_enrichment": True, "metadata_missing": ["enrichment failed"], "enrichment_error": str(exc)}
        if body.get("apply"):
            db.upsert_track(enriched)
        self._json(200, {"track": enriched})

    def _train_status(self):
        if not GH_TOKEN:
            self._json(503, {"error": "training is not configured (GH_WORKFLOW_TOKEN missing)"})
            return
        try:
            run = latest_train_run()
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"github: {exc}"})
            return
        self._json(200, {"run": run_summary(run)})

    def _trigger_train(self):
        if not GH_TOKEN:
            self._json(503, {"error": "training is not configured (GH_WORKFLOW_TOKEN missing)"})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}
        try:
            run = latest_train_run()
            if run and run.get("status") in ("queued", "in_progress", "waiting", "requested", "pending"):
                self._json(409, {"error": "a training run is already in progress", "run": run_summary(run)})
                return
            inputs = {
                "name": str(body.get("name") or "")[:80],
                "major": "true" if body.get("major") else "false",
                "notes": str(body.get("notes") or "")[:500],
            }
            resp = requests.post(
                f"{GH_API}/repos/{GH_REPO}/actions/workflows/{GH_WORKFLOW}/dispatches",
                headers=gh_headers(),
                json={"ref": "main", "inputs": inputs},
                timeout=20,
            )
            if resp.status_code not in (201, 204):
                self._json(502, {"error": f"github dispatch failed ({resp.status_code}): {resp.text[:200]}"})
                return
            # Give GitHub a moment, then report the queued run.
            time.sleep(1.5)
            run = latest_train_run()
            self._json(202, {"ok": True, "run": run_summary(run)})
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": f"github: {exc}"})

    def _unrate(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        title = (body.get("title") or "").strip()
        artist = (body.get("artist") or "").strip()
        if not title or not artist:
            self._json(400, {"error": "title and artist are required"})
            return
        n = db.delete_track(title, artist)
        self._json(200, {"ok": True, "deleted": n})

    def _add_pending(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            review = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        if not (review.get("title") or "").strip() or not (review.get("artist") or "").strip():
            self._json(400, {"error": "title and artist are required"})
            return
        try:
            review = enrich.enrich_track(review, db.fetch_tracks(), get_token())
        except Exception:  # noqa: BLE001
            review["needs_enrichment"] = True
        pid = db.add_pending_review(review)
        self._json(200, {"ok": True, "id": pid})

    def _clear_pending(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                body = {}
        ids = body.get("ids") or None
        n = db.clear_pending_reviews(ids)
        self._json(200, {"ok": True, "cleared": n})

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
        try:
            merged = enrich.enrich_track(merged, db.fetch_tracks(), get_token())
        except Exception:  # noqa: BLE001
            merged["needs_enrichment"] = True
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
        all_tracks = db.fetch_tracks()
        existing = db.match_review(candidate["title"], candidate["artist"], all_tracks)
        if not existing:
            existing = next((t for t in all_tracks if t.get("spotify_id") and t.get("spotify_id") == track["id"]), None)
        existing = existing or {}
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
            "reviewed": bool(existing),
            "rating": existing.get("rating"),
            "notes": existing.get("notes"),
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
        all_tracks = db.fetch_tracks()

        track_rows = []
        for t, c, s in zip(tracks, candidates, scores):
            existing = db.match_review(c["title"], c["artist"], all_tracks) or {}
            track_rows.append({
                "title": c["title"],
                "artist": c["artist"],
                "score": s,
                "image": track_image(t),
                "url": track_url(t),
                "spotify_id": t["id"],
                "reviewed": bool(existing),
                "rating": existing.get("rating"),
                "notes": existing.get("notes"),
            })

        self._json(200, {
            "name": meta.get("name"),
            "owner": (meta.get("owner") or {}).get("display_name"),
            "image": (meta.get("images") or [{}])[0].get("url") if meta.get("images") else None,
            "url": (meta.get("external_urls") or {}).get("spotify"),
            "tracks": track_rows,
        })

    def _album(self, params):
        q = (params.get("q") or [""])[0].strip()
        aid = (params.get("id") or [""])[0].strip()
        token = get_token()

        if q and not aid:
            albums = search_albums(token, q)
            all_tracks = db.fetch_tracks()
            out = []
            for a in albums:
                summary = album_summary(a)
                existing = db.match_review(summary["name"], summary["artist"], all_tracks) or {}
                summary["reviewed"] = bool(existing)
                summary["rating"] = existing.get("rating")
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
        all_tracks = db.fetch_tracks()

        track_rows = []
        for t, c, s in zip(tracks, candidates, scores):
            existing = db.match_review(c["title"], c["artist"], all_tracks) or {}
            track_rows.append({
                "title": c["title"],
                "artist": c["artist"],
                "score": s,
                "image": album_image,
                "url": track_url(t),
                "spotify_id": t["id"],
                "reviewed": bool(existing),
                "rating": existing.get("rating"),
                "notes": existing.get("notes"),
            })

        self._json(200, {
            "name": meta.get("name"),
            "artist": album_artist,
            "year": year,
            "image": album_image,
            "url": (meta.get("external_urls") or {}).get("spotify"),
            "album_id": aid,
            "reviewed": db.is_reviewed(meta.get("name"), album_artist),
            "tracks": track_rows,
        })

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
