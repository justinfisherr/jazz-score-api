#!/usr/bin/env python3
"""
Score a candidate jazz track against the trained taste model.

Scoring reuses the model's OWN feature pipeline (train.engineer_features), so the
Bayesian collaborator/era/label averages the model actually depends on are computed
exactly as they were at train time. A candidate that already exists in the dataset
is scored in place; a genuinely new candidate is appended unrated. Both paths see
the same encodings (self and same-album siblings excluded), so the score does not
depend on where the candidate was inserted.

Usage:
  Interactive:  python3 predict.py
  JSON input:   python3 predict.py --json '{"title":"Naima","artist":"John Coltrane",...}'
  From file:    python3 predict.py --file candidates.json
"""

import json
import sys
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import joblib

import train  # reuse the exact feature engineering the model was trained on

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "model.joblib"
VERSIONS_DIR = SCRIPT_DIR / "versions"
TRAINING_PATH = SCRIPT_DIR / "training-data.json"


@lru_cache(maxsize=8)
def _load_model_file(path_str):
    return joblib.load(path_str)


def load_model(version=None):
    if version:
        model_path = VERSIONS_DIR / f"v{version}" / "model.joblib"
    else:
        model_path = MODEL_PATH
    if not model_path.exists():
        print(f"No model found at {model_path}. Run `python3 train.py` first.")
        sys.exit(1)
    # Cached: the recommender scores a batch of candidates per request and each
    # predict_track() used to reload the joblib from disk.
    return _load_model_file(str(model_path))


def _load_training_tracks():
    # Prefer the live Postgres dataset (when DATABASE_URL is set), else fall back
    # to the local JSON snapshot. Mirrors the album filter used at train time.
    tracks = _load_from_db() or _load_from_json()
    return [t for t in tracks if t.get("entity_type") != "album"]


def _load_from_db():
    try:
        import db  # noqa: PLC0415

        return db.fetch_tracks() or None
    except Exception:  # noqa: BLE001
        return None


def _load_from_json():
    data = json.loads(TRAINING_PATH.read_text())
    data = data if isinstance(data, list) else data.get("tracks", data.get("data", []))
    return data


def _norm(s):
    return (s or "").strip().lower()


def _match_index(candidate, tracks):
    """Return the index of an existing dataset track matching this candidate
    (title + artist), or None if it's a new track."""
    ct, ca = _norm(candidate.get("title")), _norm(candidate.get("artist"))
    if not ct:
        return None
    for i, t in enumerate(tracks):
        if _norm(t.get("title")) == ct and _norm(t.get("artist")) == ca:
            return i
    return None


def _norm_lookup(value, vocabulary):
    """Map free-text metadata onto the vocabulary the model was trained on."""
    if not value:
        return None
    v = str(value).strip().lower()
    exact = {str(x).lower(): x for x in vocabulary or []}
    if v in exact:
        return exact[v]
    for k, original in exact.items():
        if v in k or k in v:
            return original
    return None


def normalize_candidate(track, model_data):
    """Coerce a candidate's tags into the exact values train.py saw.

    The live recommender supplies tags from a language model, not from the
    curated dataset, so "hard bop" / "tenor sax" / "medium fast" would miss
    every one-hot column and every group-mean lookup. This maps them onto the
    trained vocabulary and fills instrumentation from key_players when absent,
    so the score reflects the track instead of a wall of fallback defaults.
    """
    c = dict(track)

    era = _norm_lookup(c.get("era"), model_data.get("eras"))
    c["era"] = era or "Unknown"

    instrument = str(c.get("primary_instrument") or "").strip().lower()
    instrument = {
        "tenor sax": "tenor saxophone",
        "alto sax": "alto saxophone",
        "soprano sax": "soprano saxophone",
        "bari sax": "baritone saxophone",
        "baritone sax": "baritone saxophone",
        "sax": "saxophone",
        "acoustic bass": "bass",
        "upright bass": "bass",
        "double bass": "bass",
        "drums": "drums",
        "voice": "vocals",
    }.get(instrument, instrument)
    c["primary_instrument"] = instrument or "Unknown"

    if isinstance(c.get("subgenres"), list):
        common = model_data.get("common_subgenres") or []
        mapped = []
        for sg in c["subgenres"]:
            hit = _norm_lookup(sg, common)
            mapped.append(hit or str(sg).strip().lower())
        c["subgenres"] = sorted(set(m for m in mapped if m))

    tempo = str(c.get("tempo") or "").strip().lower().replace("-", " ")
    c["tempo"] = {
        "medium fast": "medium-fast",
        "mid": "medium",
        "mid tempo": "medium",
        "midtempo": "medium",
        "up": "fast",
        "up tempo": "fast",
        "uptempo": "fast",
        "ballad": "slow",
    }.get(tempo, tempo.replace(" ", "-"))
    if c["tempo"] not in train.TEMPO_MAP:
        c["tempo"] = "medium"

    complexity = str(c.get("harmonic_complexity") or "").strip().lower()
    c["harmonic_complexity"] = complexity if complexity in train.COMPLEXITY_MAP else "medium"

    if not c.get("instrumentation"):
        derived = []
        for kp in c.get("key_players") or []:
            if isinstance(kp, str) and " - " in kp:
                derived.append(kp.split(" - ", 1)[1].strip().lower())
        if derived:
            c["instrumentation"] = sorted(set(derived))

    return c


def predict_tracks(candidates, version=None):
    """Score a list of candidate track dicts. Returns a list of floats (1-10),
    aligned with the input order.

    Candidates are appended unrated. Every target encoding in train.py skips
    unrated rows and ignores same-album siblings, so a candidate's features are
    the same whether it already exists in the dataset or not. There is no
    placeholder rating and no dependence on where the row lands in the list.
    """
    model_data = load_model(version)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]

    train_tracks = _load_training_tracks()

    combined = list(train_tracks)
    positions = []
    for c in candidates:
        idx = _match_index(c, train_tracks)
        if idx is not None:
            positions.append(idx)                     # score in place (self/siblings excluded)
        else:
            cc = normalize_candidate(c, model_data)
            cc.pop("rating", None)                    # a candidate has no rating
            positions.append(len(combined))
            combined.append(cc)

    X, *_ = train.engineer_features(combined)
    X = X.reindex(columns=feature_names, fill_value=0)  # align to the model's feature set/order
    preds = np.clip(model.predict(scaler.transform(X)), 1, 10)
    return [round(float(preds[p]), 2) for p in positions]


def predict_track(track, model_data=None, version=None):
    """Backward-compatible single-track API. `model_data` is accepted and ignored
    (the model is loaded internally to keep the feature pipeline consistent)."""
    return predict_tracks([track], version=version)[0]


def interactive_input():
    print("\n--- Score a Jazz Track ---\n")
    track = {}
    track["title"] = input("Title: ").strip()
    track["artist"] = input("Artist: ").strip()
    track["energy"] = int(input("Energy (1-10): ").strip() or "5")
    track["tempo"] = input("Tempo (slow/medium/medium-fast/fast): ").strip() or "medium"
    track["harmonic_complexity"] = input("Harmonic complexity (low/medium/high): ").strip() or "medium"
    track["era"] = input("Era (Modal/Post-Bop/Cool Jazz/Hard Bop/Bebop/Swing): ").strip()
    track["primary_instrument"] = input("Primary instrument: ").strip()
    track["year"] = int(input("Year: ").strip() or "1960")
    track["label"] = input("Label (or blank): ").strip() or None

    moods_input = input("Moods (comma-separated): ").strip()
    track["moods"] = [m.strip() for m in moods_input.split(",") if m.strip()]

    subgenres_input = input("Subgenres (comma-separated): ").strip()
    track["subgenres"] = [s.strip() for s in subgenres_input.split(",") if s.strip()]

    instr_input = input("Instrumentation (comma-separated): ").strip()
    track["instrumentation"] = [i.strip() for i in instr_input.split(",") if i.strip()]

    players_input = input("Key players (comma-separated): ").strip()
    track["key_players"] = [p.strip() for p in players_input.split(",") if p.strip()]

    return track


def display_result(track, score):
    title = track.get("title", "Unknown")
    artist = track.get("artist", "Unknown")

    if score >= 8.5:
        verdict = "STRONG RECOMMEND"
    elif score >= 7:
        verdict = "RECOMMEND"
    elif score >= 5.5:
        verdict = "MAYBE"
    else:
        verdict = "SKIP"

    print(f"\n  {title} — {artist}")
    print(f"  Predicted rating: {score}/10")
    print(f"  Verdict: {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Score jazz tracks against your taste model")
    parser.add_argument("--json", help="JSON string of a single track")
    parser.add_argument("--file", help="Path to JSON file with array of candidate tracks")
    parser.add_argument("--version", help="Load model from a specific major version (e.g., '1.00')")
    args = parser.parse_args()

    if args.json:
        track = json.loads(args.json)
        score = predict_track(track, version=args.version)
        display_result(track, score)

    elif args.file:
        tracks = json.loads(Path(args.file).read_text())
        if isinstance(tracks, dict):
            tracks = [tracks]
        print(f"\nScoring {len(tracks)} tracks...\n")
        scores = predict_tracks(tracks, version=args.version)
        results = list(zip(tracks, scores))
        for t, s in results:
            display_result(t, s)

        results.sort(key=lambda x: x[1], reverse=True)
        print("--- Ranked ---")
        for t, s in results:
            print(f"  {s}/10  {t.get('title', '?')} — {t.get('artist', '?')}")

    else:
        track = interactive_input()
        score = predict_track(track, version=args.version)
        display_result(track, score)


if __name__ == "__main__":
    main()
