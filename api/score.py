"""Vercel serverless function: search Spotify + score a track against the taste model.

GET /api/score?q=moanin%20art%20blakey
GET /api/score?title=So%20What&artist=Miles%20Davis
"""

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from score import (  # noqa: E402
    build_candidate,
    fetch_reccobeats,
    fetch_spotify_audio,
    get_token,
    search_track,
)
from predict import predict_track  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        params = parse_qs(urlparse(self.path).query)
        q = (params.get("q") or [""])[0].strip()
        title = (params.get("title") or [""])[0].strip()
        artist = (params.get("artist") or [""])[0].strip()

        if not q and not (title and artist):
            self._json(400, {"error": "provide q or title+artist"})
            return

        try:
            token = get_token()
            track = search_track(token, q, title or None, artist or None)
            if not track:
                self._json(404, {"error": "no track found"})
                return
            sfa = fetch_spotify_audio(token, track["id"])
            recco = fetch_reccobeats(track["id"])
            candidate = build_candidate(track, sfa, recco)
            score = predict_track(candidate)
            self._json(200, {
                "title": candidate["title"],
                "artist": candidate["artist"],
                "year": candidate["year"],
                "score": score,
                "tempo": candidate["tempo"],
                "popularity": candidate["audio_features"].get("popularity"),
                "audio_only": True,
            })
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc)})

    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
