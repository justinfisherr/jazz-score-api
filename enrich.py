"""Metadata enrichment for newly-rated tracks.

The app rating paths only collect title/artist/rating (+ optional spotify_id).
This module fills the fields the model actually needs (era, primary instrument,
label, key players, instrumentation, subgenres, moods) by combining Spotify
track metadata with patterns already present in the dataset.

Anything that cannot be inferred confidently is left empty and flagged with
`needs_enrichment: true`, so the dashboard/training can surface it instead of
silently treating an empty list as complete.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import requests

SPOTIFY_API = "https://api.spotify.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
ERA_OPTIONS = {
    "Afro-Cuban Jazz", "Bebop", "Bossa Nova", "Contemporary", "Cool Jazz", "Cool",
    "ECM/Improvisation", "Electric", "Free Jazz", "Hard Bop", "Jazz Fusion", "Modal",
    "Neo-Bop", "Post-Bop", "Soul/Funk", "Spiritual Jazz", "Swing", "Vocal Jazz", "Unknown",
}


def norm(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def norm_title(value) -> str:
    s = norm(value)
    s = re.sub(r"\s*[-–]\s*(19|20)\d\d\s+.*$", "", s)
    s = re.sub(r"\s*[\(\[].*?(?:remaster|mono|stereo|take \d|alternate|live|version|edition).*?[\)\]]", "", s)
    s = re.sub(r"\s*-\s*(remaster(ed)?|mono|stereo|live).*$", "", s)
    return s.strip()


def _artist_ok(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return bool(a and b and (a == b or a in b or b in a))


def _mode(values):
    clean = [v for v in values if v]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _same_recording(track, ref) -> bool:
    return norm_title(track.get("title")) == norm_title(ref.get("title")) and _artist_ok(track.get("artist"), ref.get("artist"))


def _same_album(track, ref) -> bool:
    a, b = norm(track.get("album")), norm(ref.get("album"))
    return bool(a and b and a == b)


def reference_tracks(track, tracks):
    exact = [t for t in tracks if _same_recording(track, t)]
    album = [t for t in tracks if norm(track.get("album")) and _same_album(track, t)]
    artist = [t for t in tracks if _artist_ok(track.get("artist"), t.get("artist"))]
    return exact, album, artist


def era_from_year(year):
    if not year:
        return None
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    if y < 1950:
        return "Swing"
    if y < 1955:
        return "Bebop"
    if y < 1960:
        return "Hard Bop"
    if y < 1965:
        return "Post-Bop"
    if y < 1970:
        return "Post-Bop"
    if y < 1980:
        return "Jazz Fusion"
    return "Contemporary"


def infer_era(track, exact, album, artist, all_tracks):
    for group in (exact, album, artist):
        value = _mode([t.get("era") for t in group])
        if value:
            return value, "dataset"
    year = track.get("year")
    if year:
        # Prefer the most common era among enriched tracks released near this year.
        y = int(year)
        near = [t for t in all_tracks if t.get("era") and t.get("year") and abs(int(t["year"]) - y) <= 3]
        value = _mode([t.get("era") for t in near])
        if value:
            return value, "year-neighbours"
        return era_from_year(year), "year-rule"
    return None, None


def fetch_spotify_track(token, track):
    """Return (spotify_track_dict, album_dict) or (None, None)."""
    sid = (track.get("spotify_id") or "").strip()
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if sid:
        try:
            r = requests.get(f"{SPOTIFY_API}/tracks/{sid}", headers=headers, timeout=20)
            if r.status_code == 200:
                data = r.json()
        except requests.RequestException:
            pass
    if data is None:
        q = f"{track.get('title', '')} {track.get('artist', '')}".strip()
        if not q:
            return None, None
        try:
            r = requests.get(f"{SPOTIFY_API}/search", headers=headers, params={"q": q, "type": "track", "limit": 5}, timeout=20)
            r.raise_for_status()
            items = r.json().get("tracks", {}).get("items", [])
            for item in items:
                if _artist_ok(track.get("artist"), ", ".join(a.get("name", "") for a in item.get("artists", []))):
                    data = item
                    break
        except requests.RequestException:
            return None, None
    if not data:
        return None, None
    album = data.get("album") or {}
    return data, album


def agent_metadata(track):
    """Ask the model for facts Spotify does not expose.

    Only fills fields we are missing, and only accepts medium/high confidence.
    The source is recorded so curated/dataset data always wins.
    """
    if not OPENROUTER_API_KEY:
        return None
    prompt = (
        "You are a jazz discography editor. Fill this recording's metadata from well-established facts. "
        "Use the ORIGINAL release year, not a reissue year. Do not invent personnel; use [] if unsure. "
        "Era must be one of: " + ", ".join(sorted(ERA_OPTIONS)) + ". "
        "Return JSON only: "
        '{"year":1968,"era":"Post-Bop","primary_instrument":"tenor saxophone","label":"Atlantic",'
        '"key_players":["Name - instrument"],"subgenres":["post-bop"],"moods":["bluesy"],'
        '"confidence":"high|medium|low","notes":"one short source note"}'
    )
    payload = {
        "title": track.get("title"),
        "artist": track.get("artist"),
        "album": track.get("album"),
        "year": track.get("year"),
        "spotify_id": track.get("spotify_id"),
    }
    try:
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
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0.2,
                "max_tokens": 900,
                "reasoning": {"enabled": False},
            },
            timeout=45,
        )
        if resp.status_code >= 400:
            return None
        content = resp.json()["choices"][0]["message"].get("content")
        if not content:
            return None
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.S)
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        if str(data.get("confidence", "")).lower() not in ("high", "medium"):
            return None
        if data.get("era") and data["era"] not in ERA_OPTIONS:
            data.pop("era")
        if data.get("primary_instrument") and len(str(data["primary_instrument"])) > 60:
            data.pop("primary_instrument")
        if data.get("label") and len(str(data["label"])) > 80:
            data.pop("label")
        players = data.get("key_players")
        if isinstance(players, list):
            data["key_players"] = [p for p in players if isinstance(p, str) and " - " in p][:20]
        else:
            data.pop("key_players", None)
        for key in ("subgenres", "moods"):
            if not isinstance(data.get(key), list):
                data.pop(key, None)
        if data.get("year") is not None:
            try:
                y = int(data["year"])
                if 1900 <= y <= 2030:
                    data["year"] = y
                else:
                    data.pop("year")
            except (TypeError, ValueError):
                data.pop("year")
        return data
    except Exception:  # noqa: BLE001
        return None


def enrich_track(track, all_tracks, token, spotify_enabled=True, use_agent=True):
    """Fill model-critical metadata. User-supplied values are never overwritten."""
    out = dict(track)
    fill = dict(track)

    if spotify_enabled and (not out.get("album") or not out.get("year") or not out.get("spotify_id")):
        sp, album = fetch_spotify_track(token, out)
        if sp:
            fill["spotify_id"] = out.get("spotify_id") or sp.get("id")
            fill["album"] = out.get("album") or album.get("name")
            release = album.get("release_date") or ""
            if not out.get("year") and release[:4].isdigit():
                fill["year"] = int(release[:4])
                out["year_source"] = "spotify-release"
            if not out.get("album") and album.get("name"):
                out["album_source"] = "spotify"

    for key in ("spotify_id", "album", "year"):
        if not out.get(key) and fill.get(key):
            out[key] = fill[key]

    exact, album_refs, artist_refs = reference_tracks(out, all_tracks)
    sources = []

    # Sessions are specific: only copy personnel, label and instrumentation
    # from the same recording or the same album, never from an unrelated
    # session by the same artist.
    for group_name, group in (("recording", exact), ("album", album_refs)):
        if not group:
            continue
        if not out.get("primary_instrument"):
            value = _mode([t.get("primary_instrument") for t in group])
            if value:
                out["primary_instrument"] = value
                sources.append(f"instrument:{group_name}")
        if not out.get("label"):
            value = _mode([t.get("label") for t in group])
            if value:
                out["label"] = value
                sources.append(f"label:{group_name}")
        if not out.get("instrumentation"):
            value = _mode([tuple(t.get("instrumentation") or []) for t in group])
            if value:
                out["instrumentation"] = list(value)
                sources.append(f"instrumentation:{group_name}")
        if not out.get("key_players"):
            value = _mode([tuple(t.get("key_players") or []) for t in group])
            if value:
                out["key_players"] = list(value)
                sources.append(f"players:{group_name}")
        if not out.get("subgenres"):
            value = _mode([tuple(t.get("subgenres") or []) for t in group])
            if value:
                out["subgenres"] = list(value)
                sources.append(f"subgenres:{group_name}")
        if not out.get("moods"):
            value = _mode([tuple(t.get("moods") or []) for t in group])
            if value:
                out["moods"] = list(value)
                sources.append(f"moods:{group_name}")
        if out.get("year_source") == "spotify-release" or not out.get("year"):
            value = _mode([t.get("year") for t in group])
            if value:
                out["year"] = value
                out["year_source"] = f"dataset-{group_name}"
                sources.append(f"year:{group_name}")

    # Instrument, era, subgenre and mood can safely lean on the artist's
    # broader catalog when the exact session is unknown.
    if artist_refs:
        if not out.get("primary_instrument"):
            value = _mode([t.get("primary_instrument") for t in artist_refs])
            if value:
                out["primary_instrument"] = value
                sources.append("instrument:artist")
        if not out.get("subgenres"):
            value = _mode([tuple(t.get("subgenres") or []) for t in artist_refs])
            if value:
                out["subgenres"] = list(value)
                sources.append("subgenres:artist")
        if not out.get("moods"):
            value = _mode([tuple(t.get("moods") or []) for t in artist_refs])
            if value:
                out["moods"] = list(value)
                sources.append("moods:artist")

    if not out.get("era"):
        value, source = infer_era(out, exact, album_refs, artist_refs, all_tracks)
        if value:
            out["era"] = value
            out["era_source"] = source
            sources.append(f"era:{source}")

    # Spotify/older rows often cannot supply instrument, personnel, or the
    # original release year. Ask the model only for what is still missing.
    missing = [k for k in ("era", "primary_instrument", "label", "key_players", "subgenres", "moods") if not out.get(k)]
    if use_agent and missing:
        meta = agent_metadata(out)
        if meta:
            if meta.get("era") and (not out.get("era") or out.get("era_source") in ("year-neighbours", "year-rule")):
                out["era"] = meta["era"]
                sources.append("era:agent")
                out.pop("era_source", None)
            for key in ("primary_instrument", "label", "key_players", "subgenres", "moods"):
                if not out.get(key) and meta.get(key):
                    out[key] = meta[key]
                    sources.append(f"{key}:agent")
            if meta.get("year") and (out.get("year_source") == "spotify-release" or not out.get("year")):
                out["year"] = meta["year"]
                sources.append("year:agent")
                out.pop("year_source", None)
            out["agent_confidence"] = meta.get("confidence")
            if meta.get("notes"):
                out["agent_notes"] = meta["notes"]

    # Normalize the fields the model expects to exist.
    out["key_players"] = out.get("key_players") or []
    out["instrumentation"] = out.get("instrumentation") or []
    out["subgenres"] = out.get("subgenres") or []
    out["moods"] = out.get("moods") or []
    if sources:
        out["metadata_sources"] = sorted(set(sources))

    critical_missing = [k for k in ("era", "primary_instrument") if not out.get(k)]
    soft_missing = [k for k in ("label", "key_players", "subgenres", "moods") if not out.get(k)]
    out["needs_enrichment"] = bool(critical_missing)
    out["metadata_complete"] = not critical_missing and not soft_missing
    if critical_missing:
        out["metadata_missing"] = critical_missing + soft_missing
    return out
