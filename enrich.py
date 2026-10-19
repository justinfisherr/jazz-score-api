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

import re
from collections import Counter

import requests

SPOTIFY_API = "https://api.spotify.com/v1"


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


def enrich_track(track, all_tracks, token, spotify_enabled=True):
    """Fill model-critical metadata. User-supplied values are never overwritten."""
    out = dict(track)
    fill = dict(track)

    if spotify_enabled and (not out.get("album") or not out.get("year") or not out.get("spotify_id")):
        sp, album = fetch_spotify_track(token, out)
        if sp:
            fill["spotify_id"] = out.get("spotify_id") or sp.get("id")
            fill["album"] = out.get("album") or album.get("name")
            release = album.get("release_date") or ""
            fill["year"] = out.get("year") or (int(release[:4]) if release[:4].isdigit() else None)

    for key in ("spotify_id", "album", "year"):
        if not out.get(key) and fill.get(key):
            out[key] = fill[key]

    exact, album_refs, artist_refs = reference_tracks(out, all_tracks)
    sources = []
    for group_name, group in (("recording", exact), ("album", album_refs), ("artist", artist_refs)):
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
        if not out.get("key_players"):
            value = _mode([tuple(t.get("key_players") or []) for t in group])
            if value:
                out["key_players"] = list(value)
                sources.append(f"players:{group_name}")

    if not out.get("era"):
        value, source = infer_era(out, exact, album_refs, artist_refs, all_tracks)
        if value:
            out["era"] = value
            sources.append(f"era:{source}")

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
