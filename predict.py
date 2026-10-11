#!/usr/bin/env python3
"""
Score a candidate jazz track against the trained taste model.

Scoring reuses the model's OWN feature pipeline (train.engineer_features), so the
Bayesian collaborator/era/label averages the model actually depends on are computed
exactly as they were at train time. A candidate that already exists in the dataset
is scored in place (leave-one-out); a genuinely new candidate is appended with a
neutral placeholder rating and scored via the same LOO encoding.

Usage:
  Interactive:  python3 predict.py
  JSON input:   python3 predict.py --json '{"title":"Naima","artist":"John Coltrane",...}'
  From file:    python3 predict.py --file candidates.json
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import joblib

import train  # reuse the exact feature engineering the model was trained on

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "model.joblib"
VERSIONS_DIR = SCRIPT_DIR / "versions"
TRAINING_PATH = SCRIPT_DIR / "training-data.json"


def load_model(version=None):
    if version:
        model_path = VERSIONS_DIR / f"v{version}" / "model.joblib"
    else:
        model_path = MODEL_PATH
    if not model_path.exists():
        print(f"No model found at {model_path}. Run `python3 train.py` first.")
        sys.exit(1)
    return joblib.load(model_path)


def _load_training_tracks():
    data = json.loads(TRAINING_PATH.read_text())
    data = data if isinstance(data, list) else data.get("tracks", data.get("data", []))
    # Exclude album entities — they carry null era/energy and are not part of the model
    # (mirrors train.load_data). Without this, feature engineering crashes on their nulls.
    return [t for t in data if t.get("entity_type") != "album"]


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


def predict_tracks(candidates, version=None):
    """Score a list of candidate track dicts. Returns a list of floats (1-10),
    aligned with the input order."""
    model_data = load_model(version)
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]

    train_tracks = _load_training_tracks()
    rated = [t["rating"] for t in train_tracks if t.get("rating") is not None]
    global_mean = int(round(float(np.mean(rated)))) if rated else 6

    combined = list(train_tracks)
    positions = []
    for c in candidates:
        idx = _match_index(c, train_tracks)
        if idx is not None:
            positions.append(idx)                     # score in place (LOO excludes self)
        else:
            cc = dict(c)
            cc.setdefault("rating", global_mean)       # neutral placeholder; LOO ignores it for its own row
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
