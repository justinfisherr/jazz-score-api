#!/usr/bin/env python3
"""
Search Spotify for a track, pull its audio profile, and score it against the
trained jazz taste model.

    python3 score.py "naima coltrane"                 # free-text search, top hit
    python3 score.py --title Naima --artist "John Coltrane"

Spotify credentials come from ~/spotify-mcp-server/spotify-config.json (client
credentials) or SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET env vars.

NOTE: this scores the *audio profile* only. The model's jazz tags (moods,
subgenres, instrumentation, era, key players) aren't in Spotify, so the prediction
is a rough "do I like how this sounds" estimate, not a full recommendation.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path.home() / "spotify-mcp-server" / "spotify-config.json"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API = "https://api.spotify.com/v1"
RECCOBEATS_URL = "https://api.reccobeats.com/v1/audio-features"


def _creds():
    cid = os.environ.get("SPOTIFY_CLIENT_ID")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET")
    if cid and secret:
        return cid, secret
    if not CONFIG_PATH.exists():
        raise SystemExit(f"No Spotify credentials. Set SPOTIFY_CLIENT_ID/SECRET or create {CONFIG_PATH}")
    cfg = json.loads(CONFIG_PATH.read_text())
    return cfg["clientId"], cfg["clientSecret"]


def get_token():
    cid, secret = _creds()
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        headers={"Authorization": f"Basic {auth}"},
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_track(token, query, title=None, artist=None):
    if title and artist:
        q = f"track:{title} artist:{artist}"
    else:
        q = query
    resp = requests.get(
        f"{SPOTIFY_API}/search",
        headers={"Authorization": f"Bearer {token}"},
        params={"q": q, "type": "track", "limit": 1},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("tracks", {}).get("items", [])
    return items[0] if items else None


def fetch_spotify_audio(token, sid):
    resp = requests.get(
        f"{SPOTIFY_API}/audio-features/{sid}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code != 200:
        return {}
    return resp.json()


def fetch_reccobeats(sid):
    resp = requests.get(RECCOBEATS_URL, params={"ids": sid}, timeout=30)
    if resp.status_code != 200:
        return None
    for item in resp.json().get("content", []):
        href = item.get("href", "")
        got = href.rstrip("/").split("/")[-1] if href else None
        if got == sid:
            return {
                "acousticness": item.get("acousticness"),
                "danceability": item.get("danceability"),
                "spotify_energy": item.get("energy"),
                "instrumentalness": item.get("instrumentalness"),
                "liveness": item.get("liveness"),
                "loudness": item.get("loudness"),
                "speechiness": item.get("speechiness"),
                "spotify_valence": item.get("valence"),
            }
    return None


def tempo_label(bpm):
    if bpm is None:
        return "medium"
    if bpm < 85:
        return "slow"
    if bpm < 120:
        return "medium"
    if bpm < 150:
        return "medium-fast"
    return "fast"


def clean_title(name):
    # "So What (feat. John Coltrane, Cannonball Adderley & Bill Evans)" -> "So What"
    return re.sub(r"\s*[\(\[].*?(?:feat\.|ft\.|with ).*?[\)\]]", "", name, flags=re.IGNORECASE).strip()


def build_candidate(track, sfa, recco):
    title = clean_title(track["name"])
    artist = track["artists"][0]["name"] if track.get("artists") else "Unknown"
    album = track.get("album", {})
    year = None
    if album.get("release_date"):
        year = int(album["release_date"][:4])

    spotify_energy = (recco or {}).get("spotify_energy")
    energy = max(1, min(10, round(spotify_energy * 10))) if spotify_energy is not None else 5
    tempo_bpm = round(sfa.get("tempo")) if sfa.get("tempo") else 120

    audio_features = {
        "duration_s": round(track.get("duration_ms", 0) / 1000),
        "popularity": track.get("popularity", 50),
        "tempo_bpm": tempo_bpm,
        "time_signature": sfa.get("time_signature") or 4,
        "is_live": False,
    }
    if recco:
        audio_features.update(recco)

    return {
        "title": title,
        "artist": artist,
        "year": year,
        "energy": energy,
        "tempo": tempo_label(tempo_bpm),
        "harmonic_complexity": "medium",
        "audio_features": audio_features,
        "era": "Unknown",
        "primary_instrument": "Unknown",
        "moods": [],
        "subgenres": [],
        "instrumentation": [],
        "key_players": [],
    }


def display(candidate, score):
    title = candidate.get("title", "Unknown")
    artist = candidate.get("artist", "Unknown")
    if score >= 8.5:
        verdict = "STRONG RECOMMEND"
    elif score >= 7:
        verdict = "RECOMMEND"
    elif score >= 5.5:
        verdict = "MAYBE"
    else:
        verdict = "SKIP"

    af = candidate.get("audio_features", {})
    print(f"\n  {title} — {artist}")
    if candidate.get("year"):
        print(f"  ({candidate['year']}, {candidate['tempo']} tempo, {af.get('popularity', '?')} popularity)")
    print(f"  Predicted rating: {score}/10  —  {verdict}")
    print("  (audio profile only — no moods/subgenres/instrumentation tags)\n")


def main():
    parser = argparse.ArgumentParser(description="Search Spotify and score a track against the taste model")
    parser.add_argument("query", nargs="?", help="Free-text search query")
    parser.add_argument("--title", help="Exact title")
    parser.add_argument("--artist", help="Exact artist")
    parser.add_argument("--version", help="Model version (e.g. '1.00')")
    args = parser.parse_args()

    if not args.query and not (args.title and args.artist):
        parser.error("provide a query, or --title and --artist")

    sys.path.insert(0, str(SCRIPT_DIR))
    from predict import predict_track  # noqa: E402

    print("Getting Spotify token...")
    token = get_token()

    print("Searching Spotify...")
    track = search_track(token, args.query, args.title, args.artist)
    if not track:
        print("No track found.")
        return 1

    sid = track["id"]
    print(f"  Found: {track['name']} — {', '.join(a['name'] for a in track['artists'])}")

    sfa = fetch_spotify_audio(token, sid)
    time.sleep(0.1)
    recco = fetch_reccobeats(sid)

    candidate = build_candidate(track, sfa, recco)
    score = predict_track(candidate, version=args.version)
    display(candidate, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
