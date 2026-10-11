import json
import re
import os
import argparse
import shutil
import joblib
import time
import urllib.request
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).parent
TRAINING_DATA_PATH = SCRIPT_DIR / "training-data.json"
OUTPUT_PATH = SCRIPT_DIR / "dashboard-data.json"
MODEL_PATH = SCRIPT_DIR / "model.joblib"
VERSIONS_PATH = SCRIPT_DIR / "versions.json"
VERSIONS_DIR = SCRIPT_DIR / "versions"

TEMPO_MAP = {"slow": 1, "medium": 2, "medium-fast": 3, "fast": 4, "varied": 2.5}
COMPLEXITY_MAP = {"low": 1, "medium": 2, "high": 3}
INSTRUMENT_GROUPS = {
    "tenor saxophone": "tenor_sax", "saxophone": "tenor_sax",
    "soprano saxophone": "tenor_sax", "alto saxophone": "tenor_sax",
    "piano": "piano",
    "vocals": "vocals",
    "bass": "bass",
    "trumpet": "trumpet",
    "guitar": "guitar",
}
INST_GROUP_VALUES = ["tenor_sax", "piano", "vocals", "bass", "trumpet", "guitar", "other"]
DISCOVERY_SOURCES = ["self", "claude-recommendation", "autoplay"]
MOOD_ZONES = ["euphoric", "tense", "introspective", "serene"]
SUBGENRE_THRESHOLD = 3
LABEL_THRESHOLD = 3


def load_data():
    all_entries = json.loads(TRAINING_DATA_PATH.read_text())
    today = date.today().isoformat()
    dirty = False
    for entry in all_entries:
        if not entry.get("date_added"):
            entry["date_added"] = today
            dirty = True
    if dirty:
        TRAINING_DATA_PATH.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False))
    tracks = [t for t in all_entries if t.get("entity_type") != "album"]
    albums = [t for t in all_entries if t.get("entity_type") == "album"]
    return tracks, albums


def compute_mood_zone(track):
    audio = track.get("audio_features") or {}
    valence = audio.get("spotify_valence")
    energy = audio.get("spotify_energy")
    if valence is None or energy is None:
        return None
    if valence >= 0.5 and energy >= 0.5:
        return "euphoric"
    elif valence < 0.5 and energy >= 0.5:
        return "tense"
    elif valence < 0.5 and energy < 0.5:
        return "introspective"
    else:
        return "serene"


def engineer_features(tracks):
    rows = []
    all_subgenres = {}
    all_labels = {}

    for t in tracks:
        for s in t.get("subgenres", []):
            all_subgenres[s] = all_subgenres.get(s, 0) + 1
        label = t.get("label")
        if label:
            all_labels[label] = all_labels.get(label, 0) + 1

    common_subgenres = sorted([s for s, c in all_subgenres.items() if c >= SUBGENRE_THRESHOLD])
    common_labels = sorted([l for l, c in all_labels.items() if c >= LABEL_THRESHOLD])

    eras = sorted(set(t.get("era", "Unknown") for t in tracks))

    artist_track_indices = {}
    for i, t in enumerate(tracks):
        artist_track_indices.setdefault(t.get("artist", "Unknown"), []).append(i)
    global_mean = np.mean([t["rating"] for t in tracks])

    has_audio = sum(1 for t in tracks if t.get("audio_features")) >= 3

    # Pre-compute artist stats and collaborator ratings for LOO encoding
    artist_ratings = {}
    for i, t in enumerate(tracks):
        artist_ratings.setdefault(t.get("artist", "Unknown"), []).append((i, t["rating"]))

    player_ratings = {}
    for t in tracks:
        for kp in t.get("key_players", []):
            name = kp.split(" - ")[0].strip().lower()
            player_ratings.setdefault(name, []).append(t["rating"])

    # Sideman-only ratings: appearances where this person is NOT the credited
    # leader (artist) of the track. Kept separate from leader-track ratings so
    # a musician's fame as a sideman doesn't leak into predicting their own
    # leader sessions (see leader_sideman_mean_rating below).
    sideman_rating_pairs = {}
    for i, t in enumerate(tracks):
        t_leader_name = t.get("artist", "").lower().split("&")[0].strip()
        for kp in t.get("key_players", []):
            name = kp.split(" - ")[0].strip().lower()
            if name and t_leader_name and name not in t_leader_name and t_leader_name not in name:
                sideman_rating_pairs.setdefault(name, []).append((i, t["rating"]))

    if sideman_rating_pairs:
        sideman_means = {p: np.mean([r for _, r in rs]) for p, rs in sideman_rating_pairs.items()}
        elite_sideman_threshold = np.percentile(list(sideman_means.values()), 75)
    else:
        elite_sideman_threshold = global_mean

    # Pre-compute artist×decade, artist×era, label×decade ratings for LOO (Feature 1, 4)
    artist_decade_ratings = {}
    artist_era_ratings = {}
    label_decade_ratings = {}
    for i, t in enumerate(tracks):
        artist = t.get("artist", "Unknown")
        yr = t.get("year") or 1960
        decade = f"{(yr // 10) * 10}s"
        era = t.get("era", "Unknown")
        label = t.get("label", "Unknown")

        key_ad = (artist, decade)
        key_ae = (artist, era)
        key_ld = (label, decade)

        artist_decade_ratings.setdefault(key_ad, []).append((i, t["rating"]))
        artist_era_ratings.setdefault(key_ae, []).append((i, t["rating"]))
        label_decade_ratings.setdefault(key_ld, []).append((i, t["rating"]))

    # Pre-compute era, instrument, instrument combo, and label ratings for LOO
    era_ratings = {}
    instrument_ratings = {}
    instrument_combo_ratings = {}
    label_ratings = {}
    for i, t in enumerate(tracks):
        era = t.get("era", "Unknown")
        primary_instrument = t.get("primary_instrument", "Unknown")
        instr_tuple = tuple(sorted(set(t.get("instrumentation", []))))
        label = t.get("label", "Unknown")

        era_ratings.setdefault(era, []).append((i, t["rating"]))
        instrument_ratings.setdefault(primary_instrument, []).append((i, t["rating"]))
        instrument_combo_ratings.setdefault(instr_tuple, []).append((i, t["rating"]))
        label_ratings.setdefault(label, []).append((i, t["rating"]))

    # Pre-compute player stats for collaborator enhancements (Feature 3)
    # Stores (track_index, rating) so per-track lookups below can do proper
    # leave-one-out exclusion instead of leaking the current track's own rating.
    player_track_count = {}
    player_rating_pairs = {}
    for i, t in enumerate(tracks):
        for kp in t.get("key_players", []):
            name = kp.split(" - ")[0].strip().lower()
            player_track_count[name] = player_track_count.get(name, 0) + 1
            player_rating_pairs.setdefault(name, []).append((i, t["rating"]))

    # Identify elite collaborators (top 25% by mean rating)
    if player_rating_pairs:
        player_means = {p: np.mean([r for _, r in rs]) for p, rs in player_rating_pairs.items()}
        elite_threshold = np.percentile(list(player_means.values()), 75)
        elite_players = {p for p, m in player_means.items() if m >= elite_threshold}
    else:
        elite_players = set()

    # Compute recent period: last N tracks (approximate by index)
    recent_threshold = max(0, len(tracks) - 40)  # last ~40 tracks

    # Feature 9: Taste drift — causal EWMA of prior ratings (span=20 tracks).
    # Computed once per track using only ratings that came BEFORE it in
    # sequence, so it's leak-free: "how positively was I rating things going
    # into this track," independent of category-specific priors. Lets the
    # model see taste as a moving target rather than a stationary average.
    EWMA_SPAN = 20.0
    EWMA_ALPHA = 2.0 / (EWMA_SPAN + 1.0)
    RECENCY_HALF_LIFE = 40.0  # tracks
    RECENCY_DECAY = np.log(2) / RECENCY_HALF_LIFE
    ewma_taste_levels = []
    running_ewma = global_mean
    for t in tracks:
        ewma_taste_levels.append(running_ewma)
        running_ewma = EWMA_ALPHA * t["rating"] + (1 - EWMA_ALPHA) * running_ewma

    for idx, t in enumerate(tracks):
        row = {}
        row["energy"] = t.get("energy", 5)
        row["year"] = t.get("year") or 1960
        row["tempo"] = TEMPO_MAP.get(t.get("tempo", "medium"), 2.5)
        row["harmonic_complexity"] = COMPLEXITY_MAP.get(t.get("harmonic_complexity", "medium"), 2)

        # Era one-hot
        track_era = t.get("era", "Unknown")
        for era in eras:
            row[f"era_{era}"] = 1 if track_era == era else 0

        # Decade bucket one-hot
        yr = t.get("year") or 1960
        decade = f"{(yr // 10) * 10}s"
        for d in ["1940s", "1950s", "1960s", "1970s"]:
            row[f"decade_{d}"] = 1 if decade == d else 0

        # Instrument
        instrument = t.get("primary_instrument", "other")
        group = INSTRUMENT_GROUPS.get(instrument, "other")
        for g in INST_GROUP_VALUES:
            row[f"inst_{g}"] = 1 if group == g else 0

        # Mood zones (computed from audio features)
        mood_zone = compute_mood_zone(t)
        for zone in MOOD_ZONES:
            row[f"mood_zone_{zone}"] = 1 if mood_zone == zone else 0

        # Subgenres
        track_subgenres = t.get("subgenres", [])
        for s in common_subgenres:
            row[f"subgenre_{s}"] = 1 if s in track_subgenres else 0

        row["subgenre_count"] = len(track_subgenres)

        # Ensemble / instrumentation
        instr_lower = [i.lower() for i in t.get("instrumentation", [])]
        instr_joined = " ".join(instr_lower)
        row["ensemble_size"] = len(instr_lower)
        row["has_guitar"] = int("guitar" in instr_joined)
        row["has_strings"] = int("strings" in instr_joined or "orchestra" in instr_joined)
        row["has_vocals"] = int("vocal" in instr_joined)
        row["is_pianoless"] = int("piano" not in instr_joined)
        row["has_trombone"] = int("trombone" in instr_joined)

        # Label
        track_label = t.get("label", "Unknown")
        for l in common_labels:
            row[f"label_{l}"] = 1 if track_label == l else 0

        # Key players
        row["key_player_count"] = len(t.get("key_players", []))
        artist_name = t.get("artist", "").lower().split("&")[0].strip()
        row["artist_is_leader"] = int(any(artist_name in kp.lower() for kp in t.get("key_players", [])))

        # Collaborator quality — avg rating of OTHER tracks featuring each key
        # player (proper LOO by track index). The track's own leader is
        # excluded from this pool: otherwise a musician's fame as someone
        # else's sideman leaks into predicting their own leader session
        # (e.g. Elvin Jones's Coltrane-quartet ratings inflating the score
        # predicted for his own, much weaker, "Three Card Molly" session).
        # Fallback baseline for tracks with no named sidemen distinct from the
        # leader (~41% of the dataset: solo credits or sparse key_players).
        # Falls back to the leader's own LOO mean rating rather than the flat
        # global mean, so those tracks keep real signal instead of collapsing
        # to a constant once the leader is excluded from their own collaborator pool.
        leader_loo_entries = [r for i2, r in artist_ratings.get(t.get("artist", "Unknown"), []) if i2 != idx]
        leader_fallback_mean = float(np.mean(leader_loo_entries)) if leader_loo_entries else global_mean

        collab_ratings = []
        best_collab_rating = leader_fallback_mean
        top2_collab_ratings = []
        favorite_collab_count = 0
        has_elite_collab = 0

        for kp in t.get("key_players", []):
            name = kp.split(" - ")[0].strip().lower()
            if not name:
                continue
            is_this_track_leader = name in artist_name or artist_name in name
            entries = player_rating_pairs.get(name, [])
            other = [r for i2, r in entries if i2 != idx]
            if not other:
                continue
            collab_ratings.extend(other)
            if is_this_track_leader:
                continue  # leader's own record elsewhere shouldn't inflate their own leader track
            player_mean = float(np.mean(other))
            best_collab_rating = max(best_collab_rating, player_mean)
            top2_collab_ratings.append(player_mean)
            if player_mean >= 8.0:
                favorite_collab_count += 1
            if name in elite_players:
                has_elite_collab = 1

        row["collaborator_quality"] = float(np.mean(collab_ratings)) if collab_ratings else leader_fallback_mean
        row["best_collaborator_rating"] = float(best_collab_rating)  # Feature 3
        top2_mean = np.mean(sorted(top2_collab_ratings, reverse=True)[:2]) if top2_collab_ratings else leader_fallback_mean
        row["top2_collaborator_mean"] = float(top2_mean)  # Feature 3
        row["favorite_collaborator_count"] = float(favorite_collab_count)  # Feature 3
        collab_var = float(np.std(collab_ratings)) if len(collab_ratings) > 1 else 0.0
        row["collaborator_rating_variance"] = collab_var  # Feature 3
        row["has_elite_collaborator"] = float(has_elite_collab)  # Feature 3

        # Feature 9: Leader identity — was this track's leader a beloved
        # sideman on someone else's session? Tests directly whether
        # collaborator quality transfers to leadership.
        leader_sideman_entries = [r for i2, r in sideman_rating_pairs.get(artist_name, []) if i2 != idx]
        row["leader_sideman_track_count"] = float(len(leader_sideman_entries))
        if leader_sideman_entries:
            leader_sideman_mean = float(np.mean(leader_sideman_entries))
            row["leader_was_elite_sideman"] = int(leader_sideman_mean >= elite_sideman_threshold)
        else:
            leader_sideman_mean = global_mean
            row["leader_was_elite_sideman"] = 0
        row["leader_sideman_mean_rating"] = leader_sideman_mean

        # Discovery source
        source = t.get("discovered_from", "self")
        for s in DISCOVERY_SOURCES:
            row[f"source_{s}"] = 1 if source == s else 0

        row["energy_tempo"] = row["energy"] * row["tempo"]

        # Artist stats (LOO) with Bayesian smoothing
        artist = t.get("artist", "Unknown")
        artist_entries = artist_ratings[artist]
        n = len(artist_entries)
        k = 15  # confidence constant (IMDb-style)

        if n > 1:
            other_ratings = [r for i, r in artist_entries if i != idx]
            artist_mean = np.mean(other_ratings)
            row["artist_consistency"] = float(np.std(other_ratings)) if len(other_ratings) > 1 else 0.0
        else:
            artist_mean = global_mean
            row["artist_consistency"] = 0.0

        # Bayesian smoothing: blend artist mean with global mean based on sample size
        row["artist_mean_rating"] = float((n / (n + k)) * artist_mean + (k / (n + k)) * global_mean)
        row["artist_track_count"] = float(n)

        # Feature 5: Confidence/uncertainty features
        row["artist_rating_count_log"] = float(np.log1p(n))
        row["artist_rating_variance"] = float(row["artist_consistency"])  # already computed above
        row["artist_bayes_confidence"] = float(n / (n + k))  # raw confidence measure

        # Feature 1: Artist × Decade ratings (LOO Bayesian smoothed)
        key_ad = (artist, decade)
        ad_entries = artist_decade_ratings.get(key_ad, [])
        n_ad = len(ad_entries)
        if n_ad > 1:
            ad_other = [r for i, r in ad_entries if i != idx]
            ad_mean = np.mean(ad_other) if ad_other else artist_mean
        else:
            ad_mean = artist_mean  # fallback to artist mean
        row["artist_decade_bayes_rating"] = float((n_ad / (n_ad + k)) * ad_mean + (k / (n_ad + k)) * artist_mean)

        # Feature 1: Artist × Era ratings (LOO Bayesian smoothed)
        key_ae = (artist, track_era)
        ae_entries = artist_era_ratings.get(key_ae, [])
        n_ae = len(ae_entries)
        if n_ae > 1:
            ae_other = [r for i, r in ae_entries if i != idx]
            ae_mean = np.mean(ae_other) if ae_other else artist_mean
        else:
            ae_mean = artist_mean  # fallback to artist mean
        row["artist_era_bayes_rating"] = float((n_ae / (n_ae + k)) * ae_mean + (k / (n_ae + k)) * artist_mean)

        # Feature 1: Artist recent period delta
        artist_recent_entries = [(i, r) for i, r in artist_entries if i >= recent_threshold]
        if len(artist_recent_entries) > 1 and len(artist_entries) > 1:
            recent_other = [r for i, r in artist_recent_entries if i != idx]
            overall_other = [r for i, r in artist_entries if i != idx]
            recent_mean = np.mean(recent_other) if recent_other else global_mean
            overall_mean = np.mean(overall_other) if overall_other else global_mean
            row["artist_recent_period_delta"] = float(recent_mean - overall_mean)
        else:
            row["artist_recent_period_delta"] = 0.0

        # Era rating (LOO Bayesian smoothed)
        era = t.get("era", "Unknown")
        era_entries = era_ratings.get(era, [])
        n_era = len(era_entries)
        if n_era > 1:
            era_other = [r for i, r in era_entries if i != idx]
            era_mean = np.mean(era_other) if era_other else global_mean
        else:
            era_mean = global_mean
        row["era_bayes_rating"] = float((n_era / (n_era + k)) * era_mean + (k / (n_era + k)) * global_mean)

        # Instrument rating (LOO Bayesian smoothed)
        primary_instrument = t.get("primary_instrument", "Unknown")
        instr_entries = instrument_ratings.get(primary_instrument, [])
        n_instr = len(instr_entries)
        if n_instr > 1:
            instr_other = [r for i, r in instr_entries if i != idx]
            instr_mean = np.mean(instr_other) if instr_other else global_mean
        else:
            instr_mean = global_mean
        row["instrument_bayes_rating"] = float((n_instr / (n_instr + k)) * instr_mean + (k / (n_instr + k)) * global_mean)

        # Instrument combination rating (LOO Bayesian smoothed)
        instr_tuple = tuple(sorted(set(t.get("instrumentation", []))))
        combo_entries = instrument_combo_ratings.get(instr_tuple, [])
        n_combo = len(combo_entries)
        if n_combo > 1:
            combo_other = [r for i, r in combo_entries if i != idx]
            combo_mean = np.mean(combo_other) if combo_other else global_mean
        else:
            combo_mean = global_mean
        row["instrument_combo_bayes_rating"] = float((n_combo / (n_combo + k)) * combo_mean + (k / (n_combo + k)) * global_mean)

        # Label rating (LOO Bayesian smoothed)
        label = t.get("label", "Unknown")
        label_entries = label_ratings.get(label, [])
        n_label = len(label_entries)
        if n_label > 1:
            label_other = [r for i, r in label_entries if i != idx]
            label_mean = np.mean(label_other) if label_other else global_mean
        else:
            label_mean = global_mean
        row["label_bayes_rating"] = float((n_label / (n_label + k)) * label_mean + (k / (n_label + k)) * global_mean)

        # Duration bucket
        duration = (t.get("audio_features") or {}).get("duration_s", 300)
        row["duration_short"] = int(duration < 240)
        row["duration_long"] = int(duration > 420)
        row["duration_extra_long"] = int(duration > 600)

        # Electric / acoustic / format detection
        row["is_electric"] = int(any(k in instr_joined for k in ["electric", "synth", "clavinet", "organ", "fender", "rhodes"]))
        row["is_solo_piano"] = int(instrument == "piano" and len(instr_lower) <= 2)
        row["has_horn_section"] = int(sum(1 for i in instr_lower if any(h in i for h in ["trumpet", "trombone", "sax", "cornet", "flute", "clarinet"])) >= 3)
        row["is_collaboration"] = int("&" in t.get("artist", "") or "," in t.get("artist", ""))
        row["instrumentation_diversity"] = len(set(INSTRUMENT_GROUPS.get(i, i) for i in instr_lower))

        # Artist novelty — first time hearing this artist
        row["artist_is_new"] = int(len(artist_entries) == 1)

        # Feature 6: Recency-weighted features
        row["rating_order_percentile"] = float(idx / max(len(tracks) - 1, 1))  # 0-1 scale
        if idx >= recent_threshold:
            row["is_recent"] = 1.0
        else:
            row["is_recent"] = 0.0

        # Feature 9: Taste drift — continuous recency weight (replaces the
        # hard last-40 cutoff with smooth exponential decay) and the causal
        # EWMA current-taste-level computed above.
        row["recency_weight"] = float(np.exp(-RECENCY_DECAY * (len(tracks) - 1 - idx)))
        row["ewma_taste_level"] = float(ewma_taste_levels[idx])

        # Feature 2: Ballad × Instrumental/Vocal split
        is_ballad = any(sg in ["ballad", "piano ballad", "tenor ballad", "vocal ballad"] for sg in track_subgenres)
        row["is_ballad"] = int(is_ballad)
        row["instrumental_ballad"] = int(is_ballad and row["has_vocals"] == 0)  # ballad without vocals
        row["vocal_ballad_penalty"] = int(is_ballad and row["has_vocals"] == 1)  # ballad with vocals
        if is_ballad:
            audio = t.get("audio_features") or {}
            row["ballad_acousticness"] = float(audio.get("acousticness", 0.5))
            row["ballad_low_energy"] = int(row["energy"] <= 3)
            row["ballad_high_instrumentalness"] = float(audio.get("instrumentalness", 0.5))
        else:
            row["ballad_acousticness"] = 0.0
            row["ballad_low_energy"] = 0
            row["ballad_high_instrumentalness"] = 0.0

        # Interaction features
        row["energy_x_complexity"] = row["energy"] * row["harmonic_complexity"]

        # Key/mode features
        audio_pre = t.get("audio_features") or {}
        mode = (audio_pre.get("mode") or "").lower()
        row["is_minor_key"] = int(mode in ["minor", "dorian", "blues", "phrygian"])
        row["is_dorian"] = int(mode == "dorian")

        # Feature 7: Musically meaningful interactions
        is_hard_bop = "hard bop" in track_subgenres
        is_modal = "modal jazz" in track_subgenres
        is_free_jazz = "free jazz" in track_subgenres
        is_swing = "swing" in track_subgenres
        is_big_band = "big band" in track_subgenres or "Mingus Big Band" in t.get("artist", "")

        row["hard_bop_medium_energy"] = int(is_hard_bop and 3 <= row["energy"] <= 6)
        row["hard_bop_instrumental"] = int(is_hard_bop and row["has_vocals"] == 0)
        row["modal_low_energy"] = int(is_modal and row["energy"] <= 3)
        row["free_jazz_high_complexity_penalty"] = int(is_free_jazz and row["harmonic_complexity"] >= 2)
        row["swing_big_band_penalty"] = int(is_swing and is_big_band)
        row["energy_x_instrumentalness"] = float(row["energy"] * audio_pre.get("instrumentalness", 0.5))
        row["complexity_x_instrumentalness"] = float(row["harmonic_complexity"] * audio_pre.get("instrumentalness", 0.5))
        valence = audio_pre.get("spotify_valence", 0.5)
        instr = audio_pre.get("instrumentalness", 0.5)
        row["valence_x_instrumentalness"] = float(valence * instr)
        row["energy_x_ballad"] = float(row["energy"] * row["is_ballad"])
        row["complexity_x_ballad"] = float(row["harmonic_complexity"] * row["is_ballad"])

        # Era × energy interactions for major eras
        for era_name in ["Modal", "Free Jazz", "Cool Jazz", "Bebop"]:
            is_this_era = int(track_era == era_name)
            row[f"era_{era_name}_x_energy"] = float(is_this_era * row["energy"])

        if has_audio:
            audio = t.get("audio_features") or {}
            has_recco = audio.get("acousticness") is not None

            # User-entered fields — always included when present
            row["duration_s"]     = audio.get("duration_s") or 300
            row["popularity"]     = audio.get("popularity") or 50
            row["tempo_bpm"]      = audio.get("tempo_bpm") or 120
            row["time_signature"] = audio.get("time_signature") or 4
            row["is_live"]        = int(audio.get("is_live") or False)

            # ReccoBeats fields — real values when available, neutral defaults otherwise.
            # Feature 8: Missingness indicators
            row["has_recco_features"] = int(has_recco)
            row["missing_energy"] = int(not has_recco)
            row["missing_valence"] = int(not has_recco)
            row["missing_acousticness"] = int(not has_recco)
            row["missing_instrumentalness"] = int(not has_recco)
            row["missing_audio_any"] = int(not has_recco)
            row["missing_audio_count"] = int(not has_recco) * 4  # count of missing audio fields

            row["acousticness"]       = audio["acousticness"]     if has_recco else 0.5
            row["danceability"]       = audio["danceability"]     if has_recco else 0.5
            row["spotify_energy"]     = audio["spotify_energy"]   if has_recco else 0.5
            row["instrumentalness"]   = audio["instrumentalness"] if has_recco else 0.5
            row["liveness"]           = audio["liveness"]         if has_recco else 0.15
            row["loudness"]           = audio["loudness"]         if has_recco else -10.0
            row["speechiness"]        = audio["speechiness"]      if has_recco else 0.05
            row["spotify_valence"]    = audio["spotify_valence"]  if has_recco else 0.5
            row["spotify_valence_x_energy"] = row["spotify_valence"] * row["spotify_energy"]
        else:
            # No audio data at all; mark all missing
            row["missing_energy"] = 1
            row["missing_valence"] = 1
            row["missing_acousticness"] = 1
            row["missing_instrumentalness"] = 1
            row["missing_audio_any"] = 1
            row["missing_audio_count"] = 4

        # Feature 4: Label × Decade ratings (LOO Bayesian smoothed)
        key_ld = (track_label, decade)
        ld_entries = label_decade_ratings.get(key_ld, [])
        n_ld = len(ld_entries)
        if n_ld > 1:
            ld_other = [r for i, r in ld_entries if i != idx]
            ld_mean = np.mean(ld_other) if ld_other else global_mean
            # Approximate label mean: mean of all tracks with same label
            label_tracks = [tracks[i].get("rating") for i, t in enumerate(tracks) if t.get("label") == track_label]
            label_mean = np.mean(label_tracks) if label_tracks else global_mean
        else:
            ld_mean = global_mean
            label_mean = global_mean
        row["label_decade_bayes_rating"] = float((n_ld / (n_ld + k)) * ld_mean + (k / (n_ld + k)) * label_mean)

        # Feature 4: Prestige & Blue Note era features
        is_prestige = track_label == "Prestige"
        is_blue_note = track_label == "Blue Note"
        is_impulse = track_label == "Impulse!"
        row["prestige_1950s_1960s"] = int(is_prestige and decade in ["1950s", "1960s"])
        row["blue_note_1950s_1960s"] = int(is_blue_note and decade in ["1950s", "1960s"])
        row["impulse_1960s"] = int(is_impulse and decade == "1960s")

        # Feature 5: Collaborator count log (additional confidence metric)
        row["collaborator_count_log"] = float(np.log1p(row["key_player_count"]))
        row["collaborator_bayes_confidence"] = float(min(row["key_player_count"] / (row["key_player_count"] + 5), 1.0))  # k=5 for collaborators

        rows.append(row)

    df = pd.DataFrame(rows)
    ratings = pd.Series([t["rating"] for t in tracks])
    feature_names = list(df.columns)

    return df, ratings, feature_names, common_subgenres, common_labels


def readable_name(feat):
    NAMES = {
        "energy": "Energy",
        "year": "Year",
        "tempo": "Tempo",
        "harmonic_complexity": "Harmonic Complexity",
        "subgenre_count": "Subgenre Count",
        "energy_tempo": "Energy × Tempo",
        "artist_mean_rating": "Artist Avg Rating",
        "artist_track_count": "Artist Track Count",
        "ensemble_size": "Ensemble Size",
        "has_guitar": "Has Guitar",
        "has_strings": "Has Strings",
        "has_vocals": "Has Vocals",
        "is_pianoless": "Pianoless",
        "has_trombone": "Has Trombone",
        "key_player_count": "Key Player Count",
        "spotify_valence_x_energy": "Valence × Energy",
        "replayability": "Replayability",
        "has_favorite_moments": "Has Fav Moments",
        "duration_s": "Duration",
        "popularity": "Spotify Popularity",
        "tempo_bpm": "Tempo (BPM)",
        "time_signature": "Time Signature",
        "is_live": "Live Recording",
        "has_recco_features": "Has ReccoBeats Features",
        "acousticness": "Acousticness",
        "danceability": "Danceability",
        "spotify_energy": "Audio Energy (Spotify)",
        "instrumentalness": "Instrumentalness",
        "liveness": "Liveness Score",
        "loudness": "Loudness (dB)",
        "speechiness": "Speechiness",
        "spotify_valence": "Audio Valence (Spotify)",
        # Feature 1: Artist × Era/Decade
        "artist_decade_bayes_rating": "Artist Avg Rating (Decade)",
        "artist_era_bayes_rating": "Artist Avg Rating (Era)",
        "artist_recent_period_delta": "Artist Recent vs Overall Delta",
        # Category Bayesian ratings
        "era_bayes_rating": "Era Avg Rating",
        "instrument_bayes_rating": "Instrument Avg Rating",
        "instrument_combo_bayes_rating": "Instrument Combo Avg Rating",
        "label_bayes_rating": "Label Avg Rating",
        # Feature 2: Ballad splits
        "is_ballad": "Is Ballad",
        "instrumental_ballad": "Instrumental Ballad",
        "vocal_ballad_penalty": "Vocal Ballad",
        "ballad_acousticness": "Ballad Acousticness",
        "ballad_low_energy": "Ballad Low Energy",
        "ballad_high_instrumentalness": "Ballad High Instrumentalness",
        # Feature 3: Improved collaborator features
        "best_collaborator_rating": "Best Collaborator Avg Rating",
        "top2_collaborator_mean": "Top 2 Collaborators Mean",
        "favorite_collaborator_count": "Favorite Collaborator Count",
        "collaborator_rating_variance": "Collaborator Rating Variance",
        "has_elite_collaborator": "Has Elite Collaborator",
        # Feature 4: Label × Decade
        "label_decade_bayes_rating": "Label Avg Rating (Decade)",
        "prestige_1950s_1960s": "Prestige 1950s-60s",
        "blue_note_1950s_1960s": "Blue Note 1950s-60s",
        "impulse_1960s": "Impulse! 1960s",
        # Feature 5: Confidence/uncertainty
        "artist_rating_count_log": "Artist Rating Count (log)",
        "artist_rating_variance": "Artist Rating Variance",
        "artist_bayes_confidence": "Artist Bayes Confidence",
        "collaborator_count_log": "Collaborator Count (log)",
        "collaborator_bayes_confidence": "Collaborator Bayes Confidence",
        # Feature 6: Recency
        "rating_order_percentile": "Rating Order Percentile",
        "is_recent": "Is Recent",
        # Feature 9: Leader identity + taste drift
        "leader_sideman_mean_rating": "Leader's Rating as Sideman Elsewhere",
        "leader_was_elite_sideman": "Leader Was Elite Sideman",
        "leader_sideman_track_count": "Leader Sideman Track Count",
        "recency_weight": "Recency Weight (exp decay)",
        "ewma_taste_level": "EWMA Current Taste Level",
        # Feature 7: Musical interactions
        "hard_bop_medium_energy": "Hard Bop Medium Energy",
        "hard_bop_instrumental": "Hard Bop Instrumental",
        "modal_low_energy": "Modal Low Energy",
        "free_jazz_high_complexity_penalty": "Free Jazz High Complexity",
        "swing_big_band_penalty": "Swing Big Band",
        "energy_x_instrumentalness": "Energy × Instrumentalness",
        "complexity_x_instrumentalness": "Complexity × Instrumentalness",
        "valence_x_instrumentalness": "Valence × Instrumentalness",
        "energy_x_ballad": "Energy × Ballad",
        "complexity_x_ballad": "Complexity × Ballad",
        # Feature 8: Missingness indicators
        "missing_energy": "Missing Energy",
        "missing_valence": "Missing Valence",
        "missing_acousticness": "Missing Acousticness",
        "missing_instrumentalness": "Missing Instrumentalness",
        "missing_audio_any": "Missing Any Audio",
        "missing_audio_count": "Missing Audio Count",
    }
    if feat in NAMES:
        return NAMES[feat]
    if feat.startswith("mood_zone_"):
        zone = feat[10:].title()
        return f"{zone} zone"
    if feat.startswith("subgenre_"):
        return f'"{feat[9:].replace("_", " ").title()}" subgenre'
    if feat.startswith("era_") and "_x_energy" in feat:
        era_name = feat[4:feat.index("_x_energy")]
        return f"{era_name} Era × Energy"
    if feat.startswith("era_"):
        return f"{feat[4:]} era"
    if feat.startswith("inst_"):
        return f"{feat[5:].replace('_', ' ').title()} instrument"
    if feat.startswith("label_"):
        return f"{feat[6:]} label"
    if feat.startswith("source_"):
        return f'"{feat[7:]}" source'
    return feat.replace("_", " ").title()


def train_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=LeaveOneOut())
    ridge.fit(X_scaled, y)
    ridge_preds = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_scaled):
        r = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        r.fit(X_scaled[train_idx], y.iloc[train_idx])
        ridge_preds[test_idx] = r.predict(X_scaled[test_idx])

    ridge_metrics = {
        "r_squared": round(r2_score(y, ridge_preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y, ridge_preds)), 4),
        "mae": round(mean_absolute_error(y, ridge_preds), 4),
        "best_alpha": float(ridge.alpha_),
    }

    # v10 "raichu": depth-3 RF was the ceiling — after era+collaborator splits there was no
    # room for the conditional splits that separate tracks within a family (LOO 0.34 -> 0.51).
    # v10.01: GradientBoosting beats the depth-6 forest (LOO 0.51 -> 0.65). Boosting can push
    # predictions PAST the mean, so it reaches the 1s/10s the forest averaged toward the middle
    # (e.g. Afro Blue actual 1: RF pred 5.5 -> GBM 3.6). Kept under the "random_forest" key so
    # the React dashboard's JSON schema stays compatible; it is really a GBM (see algorithm field).
    rf = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, random_state=42)
    rf_preds = cross_val_predict(rf, X_scaled, y, cv=LeaveOneOut(), n_jobs=-1)
    rf.fit(X_scaled, y)

    rf_metrics = {
        "r_squared": round(r2_score(y, rf_preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y, rf_preds)), 4),
        "mae": round(mean_absolute_error(y, rf_preds), 4),
        "algorithm": "gradient_boosting",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
    }

    best = "ridge" if ridge_metrics["r_squared"] >= rf_metrics["r_squared"] else "random_forest"

    coefs = np.abs(ridge.coef_)
    ridge_importance = coefs / coefs.max() if coefs.max() > 0 else coefs
    ridge_directions = np.sign(ridge.coef_)

    rf_importance = rf.feature_importances_

    return {
        "ridge": ridge_metrics,
        "random_forest": rf_metrics,
        "best_model": best,
        "ridge_preds": ridge_preds,
        "rf_preds": rf_preds,
        "ridge_importance": ridge_importance,
        "ridge_directions": ridge_directions,
        "rf_importance": rf_importance,
        "fitted_model": rf,          # the fitted best tree-model (GBM) for persistence
        "scaler": scaler,
        "X_scaled": X_scaled,
    }


def get_feature_importance(model_results, feature_names, top_n=15):
    best = model_results["best_model"]
    if best == "ridge":
        importance = model_results["ridge_importance"]
        directions = model_results["ridge_directions"]
    else:
        importance = model_results["rf_importance"]
        directions = np.zeros(len(importance))

    indices = np.argsort(importance)[::-1][:top_n]
    return [
        {
            "feature": readable_name(feature_names[i]),
            "importance": round(float(importance[i]), 4),
            "direction": "positive" if directions[i] >= 0 else "negative",
        }
        for i in indices
    ]


def get_top_drivers(model_results, feature_names, top_n=15):
    """Get Ridge coefficients as actual rating drivers."""
    if model_results["best_model"] != "ridge":
        return []
    coefs = model_results["ridge_directions"] * model_results["ridge_importance"]
    indices = np.argsort(np.abs(coefs))[::-1][:top_n]
    return [
        {
            "feature": readable_name(feature_names[i]),
            "coefficient": round(float(coefs[i]), 4),
            "direction": "positive" if coefs[i] >= 0 else "negative",
        }
        for i in indices
    ]


def compute_correlations(X, y, feature_names, top_n=10):
    feature_rating = []
    for i, name in enumerate(feature_names):
        col = X.iloc[:, i]
        if col.std() == 0:
            continue
        r, p = pearsonr(col, y)
        feature_rating.append({
            "feature": readable_name(name),
            "correlation": round(float(r), 4),
            "p_value": round(float(p), 4),
        })
    feature_rating.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    top_features = [f["feature"] for f in feature_rating[:top_n]]
    top_indices = []
    for f in feature_rating[:top_n]:
        for i, name in enumerate(feature_names):
            if readable_name(name) == f["feature"]:
                top_indices.append(i)
                break

    cols = [y] + [X.iloc[:, i] for i in top_indices]
    labels = ["Rating"] + top_features
    matrix = np.corrcoef(np.array([c.values for c in cols]))

    return {
        "feature_rating": feature_rating,
        "heatmap": {
            "labels": labels,
            "matrix": [[round(float(v), 3) for v in row] for row in matrix],
        },
    }


def generate_cluster_name(mean_rating, top_zones, dominant_tempo, mean_energy, mean_complexity):
    """Generate descriptive cluster name based on characteristics.

    Uses zone-first logic so names reflect the actual audio/mood character
    of each cluster rather than absolute rating thresholds.
    """
    primary_zone = top_zones[0] if top_zones else None
    secondary_zone = top_zones[1] if len(top_zones) > 1 else None
    is_slow = dominant_tempo == "slow"
    is_upbeat = dominant_tempo in ["fast", "medium-fast"]
    is_high_energy = mean_energy >= 5.5
    is_low_energy = mean_energy < 3.0

    # Euphoric + upbeat/energetic → Groovy
    if primary_zone == "euphoric" or (is_upbeat and is_high_energy):
        return "Groovy"

    # Tense + high energy → Intense
    if primary_zone == "tense" and is_high_energy:
        return "Intense"

    # Very low energy + slow → Romantic (ballads, tender, lyrical)
    if is_low_energy and is_slow:
        return "Romantic"

    # Introspective/serene + slow → Contemplative
    if primary_zone in ("introspective", "serene") and is_slow:
        return "Contemplative"

    # High energy without euphoric zone → Hard Bop / driving
    if is_high_energy:
        return "Driving"

    # Serene without being slow → Cool
    if primary_zone == "serene" or secondary_zone == "serene":
        return "Cool"

    # Introspective with moderate energy
    if primary_zone == "introspective":
        return "Introspective"

    # No audio zones (tracks missing audio features), moderate energy
    if not top_zones:
        return "Classic"

    return "Funky"


def cluster_analysis(X_scaled, tracks, feature_names):
    pca_full = PCA(n_components=min(X_scaled.shape[1], X_scaled.shape[0] - 1))
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, 0.95) + 1)
    n_components = max(2, min(n_components, X_scaled.shape[0] - 1))

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_2d = PCA(n_components=2)
    coords = pca_2d.fit_transform(X_scaled)

    results = {"k_values_tested": [], "silhouette_scores": [], "inertias": []}
    best_k, best_score, best_labels = 2, -1, None

    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)

        cluster_sizes = [int(np.sum(labels == c)) for c in range(k)]
        if min(cluster_sizes) < 8:
            results["k_values_tested"].append(k)
            results["silhouette_scores"].append(0)
            results["inertias"].append(float(km.inertia_))
            continue

        score = silhouette_score(X_pca, labels)
        results["k_values_tested"].append(k)
        results["silhouette_scores"].append(round(float(score), 4))
        results["inertias"].append(round(float(km.inertia_), 2))

        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = km.fit_predict(X_pca)
        best_k = 2
        best_score = silhouette_score(X_pca, best_labels)

    results["best_k"] = best_k

    profiles = []
    for c in range(best_k):
        mask = best_labels == c
        cluster_tracks = [tracks[i] for i in range(len(tracks)) if mask[i]]
        cluster_ratings = [t["rating"] for t in cluster_tracks]

        mood_zone_counts = {}
        era_counts = {}
        tempo_counts = {}
        for t in cluster_tracks:
            zone = compute_mood_zone(t)
            if zone:
                mood_zone_counts[zone] = mood_zone_counts.get(zone, 0) + 1
            era_counts[t.get("era", "Unknown")] = era_counts.get(t.get("era", "Unknown"), 0) + 1
            tempo_counts[t.get("tempo", "medium")] = tempo_counts.get(t.get("tempo", "medium"), 0) + 1

        top_zones = sorted(mood_zone_counts, key=mood_zone_counts.get, reverse=True)[:2]
        top_eras = sorted(era_counts, key=era_counts.get, reverse=True)[:3]
        dominant_tempo = max(tempo_counts, key=tempo_counts.get)

        mean_energy = np.mean([t.get("energy", 5) for t in cluster_tracks])
        mean_complexity = np.mean([COMPLEXITY_MAP.get(t.get("harmonic_complexity", "medium"), 2) for t in cluster_tracks])
        mean_tempo = np.mean([TEMPO_MAP.get(t.get("tempo", "medium"), 2.5) for t in cluster_tracks])

        dists = np.linalg.norm(coords[mask] - coords[mask].mean(axis=0), axis=1)
        rep_indices = np.argsort(dists)[:3]
        representative = [cluster_tracks[i]["title"] for i in rep_indices]

        zone_distribution = {}
        for zone in MOOD_ZONES:
            zone_distribution[zone] = sum(1 for t in cluster_tracks if compute_mood_zone(t) == zone) / len(cluster_tracks)

        # Generate cluster name based on characteristics
        mean_rating_val = np.mean(cluster_ratings)
        cluster_name = generate_cluster_name(mean_rating_val, top_zones, dominant_tempo, mean_energy, mean_complexity)

        profiles.append({
            "id": int(c),
            "label": cluster_name,
            "size": int(mask.sum()),
            "mean_rating": round(float(mean_rating_val), 2),
            "mean_energy": round(float(mean_energy), 2),
            "top_mood_zones": top_zones,
            "top_eras": top_eras,
            "dominant_tempo": dominant_tempo,
            "mean_complexity": round(float(mean_complexity), 2),
            "representative_tracks": representative,
            "mood_zone_distribution": {zone: round(dist, 3) for zone, dist in zone_distribution.items()},
        })

    results["cluster_profiles"] = profiles

    return results, best_labels, coords


def build_predictions(tracks, model_results, best_labels, coords):
    best = model_results["best_model"]
    preds = model_results["ridge_preds"] if best == "ridge" else model_results["rf_preds"]

    results = []
    for i, t in enumerate(tracks):
        audio = t.get("audio_features") or {}
        results.append({
            "title": t["title"],
            "artist": t["artist"],
            "actual": t["rating"],
            "predicted": round(float(preds[i]), 2),
            "residual": round(float(t["rating"] - preds[i]), 2),
            "cluster": int(best_labels[i]),
            "liked": t.get("liked", True),
            "pca_x": round(float(coords[i, 0]), 4),
            "pca_y": round(float(coords[i, 1]), 4),
            "year": t.get("year"),
            "energy": t.get("energy", 5),
            "tempo": t.get("tempo", "medium"),
            "era": t.get("era", "Unknown"),
            "mood_zone": compute_mood_zone(t),
            "key": audio.get("key"),
            "mode": audio.get("mode"),
            "duration_s": audio.get("duration_s"),
            "popularity": audio.get("popularity"),
            "acousticness": audio.get("acousticness"),
            "danceability": audio.get("danceability"),
            "spotify_energy": audio.get("spotify_energy"),
            "instrumentalness": audio.get("instrumentalness"),
            "liveness": audio.get("liveness"),
            "loudness": audio.get("loudness"),
            "speechiness": audio.get("speechiness"),
            "spotify_valence": audio.get("spotify_valence"),
            "replayability": t.get("replayability"),
            "playthrough": t.get("playthrough"),
            "primary_instrument": t.get("primary_instrument"),
            "label": t.get("label"),
            "harmonic_complexity": t.get("harmonic_complexity"),
            "discovered_from": t.get("discovered_from"),
            "ensemble_size": len(t.get("instrumentation") or []),
            "is_pianoless": int("piano" not in " ".join(t.get("instrumentation") or []).lower()),
            "has_vocals": int("vocal" in " ".join(t.get("instrumentation") or []).lower()),
            "has_guitar": int("guitar" in " ".join(t.get("instrumentation") or []).lower()),
            "subgenres": t.get("subgenres", []),
            "artist_is_leader": int(any(t.get("artist", "").lower().split("&")[0].strip() in kp.lower() for kp in t.get("key_players", []))),
            "intro_grabbed": int(any(kw in (t.get("favorite_moments") or "").lower() + " " + (t.get("notes") or "").lower() + " " + " ".join(t.get("notable_qualities", [])).lower() for kw in ["intro", "right away"])),
            "early_bail": int((t.get("playthrough") or 0.75) < 0.3),
        })
    return results


def build_distributions(tracks, best_labels):
    ratings = [t["rating"] for t in tracks]
    bins = sorted(set(ratings))
    counts = [ratings.count(b) for b in bins]

    by_cluster = {}
    for i, t in enumerate(tracks):
        c = int(best_labels[i])
        by_cluster.setdefault(c, []).append(t["rating"])

    ratings_by_cluster = [
        {"cluster": c, "ratings": sorted(rs)} for c, rs in sorted(by_cluster.items())
    ]

    return {"ratings": {"bins": bins, "counts": counts}, "ratings_by_cluster": ratings_by_cluster}


def load_history():
    if OUTPUT_PATH.exists():
        try:
            data = json.loads(OUTPUT_PATH.read_text())
            return data.get("history", [])
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def load_versions():
    if VERSIONS_PATH.exists():
        return json.loads(VERSIONS_PATH.read_text())
    return {"current_version": "0.99", "versions": []}


def next_version(manifest, is_major=False):
    current = manifest["current_version"]
    major, minor = current.split(".")
    major, minor = int(major), int(minor)
    if is_major:
        return f"{major + 1}.00"
    return f"{major}.{minor + 1:02d}"


def save_version_artifacts(version_str, manifest, metrics_summary, is_major, name="", notes="", changelog=None):
    artifacts = {
        "dashboard_data": "dashboard-data.json",
        "model": "model.joblib",
    }

    if is_major:
        version_dir = VERSIONS_DIR / f"v{version_str}"
        version_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, version_dir / "dashboard-data.json")
        shutil.copy2(MODEL_PATH, version_dir / "model.joblib")
        shutil.copy2(TRAINING_DATA_PATH, version_dir / "training-data.md")
        artifacts = {
            "dashboard_data": f"versions/v{version_str}/dashboard-data.json",
            "model": f"versions/v{version_str}/model.joblib",
            "training_data": f"versions/v{version_str}/training-data.md",
        }

    major, minor = version_str.split(".")
    version_entry = {
        "version": version_str,
        "name": name,
        "major": int(major),
        "minor": int(minor),
        "run_date": datetime.now().isoformat(),
        "dataset_size": metrics_summary["dataset_size"],
        "best_model": metrics_summary["best_model"],
        "r_squared": metrics_summary["r_squared"],
        "rmse": metrics_summary["rmse"],
        "mae": metrics_summary["mae"],
        "feature_count": metrics_summary["feature_count"],
        "best_k": metrics_summary["best_k"],
        "ridge_r_squared": metrics_summary["ridge_r_squared"],
        "ridge_rmse": metrics_summary["ridge_rmse"],
        "rf_r_squared": metrics_summary["rf_r_squared"],
        "rf_rmse": metrics_summary["rf_rmse"],
        "notes": notes,
        "changelog": changelog or [],
        "is_major": is_major,
        "training_data_snapshot": is_major,
        "artifacts": artifacts,
    }

    manifest["versions"].append(version_entry)
    manifest["current_version"] = version_str
    VERSIONS_PATH.write_text(json.dumps(manifest, indent=2))
    return version_str


def extract_rating_changes(tracks, albums):
    """Extract tracks/albums where rating has been changed"""
    all_items = tracks + albums
    changes = []

    for item in all_items:
        history = item.get("rating_history")
        if history and len(history) > 1:
            old_rating = history[-2]["rating"]
            new_rating = history[-1]["rating"]
            change = new_rating - old_rating
            direction = "improved" if change > 0 else "downgraded"

            changes.append({
                "title": item.get("title"),
                "artist": item.get("artist"),
                "old_rating": old_rating,
                "new_rating": new_rating,
                "change": change,
                "direction": direction,
                "change_date": history[-1]["date"],
                "is_album": item.get("entity_type") == "album",
                "history": history,
                "times_changed": len(history) - 1,
            })

    # Sort by date descending
    changes.sort(key=lambda x: x["change_date"], reverse=True)

    # Compute trend
    by_date = {}
    for c in changes:
        date = c["change_date"]
        if date not in by_date:
            by_date[date] = {"improved": 0, "downgraded": 0}
        by_date[date][c["direction"]] += 1

    trend = [
        {"date": date, "improved": counts["improved"], "downgraded": counts["downgraded"]}
        for date, counts in sorted(by_date.items())
    ]

    return {
        "changes": changes,
        "trend": trend,
        "total_improvements": sum(1 for c in changes if c["direction"] == "improved"),
        "total_downgrades": sum(1 for c in changes if c["direction"] == "downgraded"),
    }


PLAYLISTS = [
    {"name": "My Top Jazz Songs", "id": "4ZnOuGzizWM1UIPZlhCAae", "tracks": [
        "My Favorite Things|John Coltrane",
        "Love Theme from Spartacus|Yusef Lateef",
        "So What|Miles Davis",
        "Wild Is The Wind|Nina Simone",
        "Peace Piece|Bill Evans",
        "God Bless the Child|Sonny Rollins",
        "Goodbye Pork Pie Hat|Charles Mingus",
        "Fleurette Africaine|Duke Ellington",
        "Mercy, Mercy, Mercy|Cannonball Adderley",
        "Chameleon|Herbie Hancock",
        "Blue Train|John Coltrane",
        "Tezeta|Mulatu Astatke",
        "Better Get Hit In Your Soul|Charles Mingus",
        "Moanin'|Mingus Big Band",
        "It Never Entered My Mind|Miles Davis",
        "Blue in Green|Miles Davis",
        "St. Thomas|Sonny Rollins",
        "Open Letter to Duke|Charles Mingus",
        "Strode Rode|Sonny Rollins",
        "Equinox|John Coltrane",
        "Naima|John Coltrane",
        "A Single Petal of a Rose|Duke Ellington",
        "Générique|Miles Davis",
        "Wise One|John Coltrane",
        "I Put A Spell On You|Nina Simone",
        "Little Sunflower|Freddie Hubbard",
        "Contemplation|McCoy Tyner",
        "Red Clay|Freddie Hubbard",
        "Fly With The Wind|McCoy Tyner",
        "Autumn Leaves|Cannonball Adderley",
        "Everybody Loves The Sunshine|Roy Ayers Ubiquity",
        "(Fallin' Like) Dominoes|Donald Byrd",
        "Prologue|Kamasi Washington",
        "All Blues|Miles Davis",
        "Turiya & Ramakrishna|Alice Coltrane",
        "Infant Eyes|Wayne Shorter",
    ]},
    {"name": "Jazz Date", "id": "6n2kibGIYXV5L3flumoiDw", "tracks": [
        "God Bless the Child|Sonny Rollins",
        "Goodbye Pork Pie Hat|Charles Mingus",
        "Love Theme from Spartacus|Yusef Lateef",
        "Fleurette Africaine|Duke Ellington",
        "It Never Entered My Mind|Miles Davis",
        "Peace Piece|Bill Evans",
        "Blue in Green|Miles Davis",
        "In a Sentimental Mood|Sonny Rollins",
        "My One and Only Love|John Coltrane",
        "Naima|John Coltrane",
        "Chelsea Bridge|Ben Webster",
        "Infant Eyes|Wayne Shorter",
        "Love Theme From The Robe|Yusef Lateef",
        "Lover Man|Charlie Parker",
        "A Single Petal of a Rose|Duke Ellington",
        "What Are You Doing the Rest of Your Life|Bill Evans",
        "Embraceable You|Barry Harris",
        "To Her Ladyship|John Coltrane",
        "'Round Midnight|McCoy Tyner",
        "My Foolish Heart|Bill Evans",
        "Where Are You|Sonny Rollins",
        "You've Changed|Dexter Gordon",
        "Europa|Gato Barbieri",
        "Misty|Sarah Vaughan",
        "Danny Boy|Ben Webster",
        "Say It Over And Over Again|John Coltrane",
        "Almost Blue|Chet Baker",
        "Stella By Starlight|Miles Davis",
        "How Long Has This Been Going On|Ben Webster",
        "Autumn Leaves|Cannonball Adderley",
        "Dear Old Stockholm|Stan Getz",
        "Weaver Of Dreams|Freddie Hubbard",
        "This Bitter Earth|Dinah Washington",
        "Light Blue|Ron Carter",
        "Dear Lord|John Coltrane",
        "Willow Weep For Me|Clifford Brown",
        "Stars Fell On Alabama|Cannonball Adderley",
        "I Fall In Love Too Easily|Miles Davis",
    ]},
    {"name": "Jazz Stank", "id": "2Dj3wiGXPwGL3RjQDa6zG5", "tracks": [
        "Better Git It in Your Soul|Charles Mingus",
        "Moanin'|Mingus Big Band",
        "Open Letter to Duke|Charles Mingus",
        "Fables of Faubus|Charles Mingus",
        "Boogie Stop Shuffle|Charles Mingus",
        "Early Summer|Ryo Fukui",
        "Moanin'|Art Blakey & The Jazz Messengers",
        "Bloomdido|Charlie Parker",
        "Yègellé Tezeta|Mulatu Astatke",
        "Haitian Fight Song|Mingus Big Band",
        "Nostalgia in Times Square|Mingus Big Band",
        "Nica's Dream|Horace Silver",
        "Red Clay|Freddie Hubbard",
        "Open Sesame|Freddie Hubbard",
        "The In Crowd|Ramsey Lewis",
        "Prologue|Kamasi Washington",
    ]},
    {"name": "Jazz Vocals", "id": "5MffriQAGUbFt2DcfkgYuU", "tracks": [
        "My Funny Valentine|Chet Baker",
        "Wild Is The Wind|Nina Simone",
        "Solitude|Billie Holiday",
        "Crazy He Calls Me|Billie Holiday",
        "Lover Man|Billie Holiday",
        "Don't Let Me Be Misunderstood|Nina Simone",
        "Tryin' Times|Roberta Flack",
        "Blue Moon|Julie London",
        "I Put A Spell On You|Nina Simone",
        "Misty|Sarah Vaughan",
        "Strange Fruit|Billie Holiday",
        "This Bitter Earth|Dinah Washington",
        "Here's To Life|Shirley Horn",
        "Cry Me A River|Julie London",
        "The Girl From Ipanema|Frank Sinatra",
    ]},
    {"name": "Jazz Pool", "id": "3cnlae8AntzgbgmzdxRZOj", "tracks": [
        "All the Things You Are|Dizzy Gillespie",
        "It Never Entered My Mind|Miles Davis",
        "Tezeta|Mulatu Astatke",
        "Wild Is The Wind|Nina Simone",
        "Solitude|Billie Holiday",
        "Crazy He Calls Me|Billie Holiday",
        "Love Theme From The Robe|Yusef Lateef",
        "Moanin'|Mingus Big Band",
        "Love Theme from Spartacus|Yusef Lateef",
        "My Favorite Things|John Coltrane",
        "Lover Man|Charlie Parker",
        "In a Sentimental Mood|Duke Ellington",
        "To Her Ladyship|John Coltrane",
        "The Bridge|Sonny Rollins",
        "In a Sentimental Mood|Sonny Rollins",
        "St. Thomas|Sonny Rollins",
        "Strode Rode|Sonny Rollins",
        "Blue Train|John Coltrane",
        "Goodbye Pork Pie Hat|Charles Mingus",
        "A Single Petal of a Rose|Duke Ellington",
        "All of Me|Charlie Parker",
        "April In Paris|Charlie Parker",
        "Infant Eyes|Wayne Shorter",
        "Peace Piece|Bill Evans",
        "So What|Miles Davis",
        "Blue in Green|Miles Davis",
        "Tryin' Times|Roberta Flack",
        "Don't Let Me Be Misunderstood|Nina Simone",
        "Lover Man|Billie Holiday",
        "Embraceable You|Barry Harris",
        "My Foolish Heart|Bill Evans",
        "Early Summer|Ryo Fukui",
        "Better Git It in Your Soul|Charles Mingus",
        "Naima|John Coltrane",
        "Equinox|John Coltrane",
        "My Funny Valentine|Chet Baker",
        "What Are You Doing the Rest of Your Life|Bill Evans",
        "In Your Own Sweet Way|Wes Montgomery",
        "Open Letter to Duke|Charles Mingus",
        "Boogie Stop Shuffle|Charles Mingus",
        "Fables of Faubus|Charles Mingus",
        "My One and Only Love|John Coltrane",
        "Moanin'|Art Blakey & The Jazz Messengers",
        "Blue Moon|Julie London",
        "Cantaloupe Island|Herbie Hancock",
        "Mercy, Mercy, Mercy|Cannonball Adderley",
        "Fleurette Africaine|Duke Ellington",
        "Waltz for Debby|Bill Evans",
        "You Don't Know What Love Is|Miles Davis",
        "Django|The Modern Jazz Quartet",
        "Chelsea Bridge|Ben Webster",
        "Chameleon|Herbie Hancock",
        "Autumn Leaves|Stan Getz",
        "John S.|Sonny Rollins",
        "Corcovado|Stan Getz",
        "Yes Or No|Wayne Shorter",
        "Everybody Loves The Sunshine|Takuya Kuroda",
        "Reflections|Kurt Rosenwinkel",
        "Willow Weep For Me|Dexter Gordon",
        "My Funny Valentine|Miles Davis",
        "Bloomdido|Charlie Parker",
        "But Not For Me|Ahmad Jamal Trio",
        "I Thought About You|Miles Davis",
        "Tenderly|Ben Webster",
        "You've Changed|Dexter Gordon",
        "Générique|Miles Davis",
        "Maiden Voyage|Herbie Hancock",
        "Europa|Gato Barbieri",
        "Yègellé Tezeta|Mulatu Astatke",
        "I Put A Spell On You|Nina Simone",
        "Blues For The Orient|Yusef Lateef",
        "Misty|Sarah Vaughan",
        "Little Sunflower|Freddie Hubbard",
        "Danny Boy|Ben Webster",
        "Haitian Fight Song|Mingus Big Band",
        "Almost Blue|Chet Baker",
        "Autumn Leaves|Ryo Fukui",
        "Stella By Starlight|Miles Davis",
        "How Long Has This Been Going On|Ben Webster",
        "Contemplation|McCoy Tyner",
        "Red Clay|Freddie Hubbard",
        "Ceora|Lee Morgan",
        "Autumn Leaves|Cannonball Adderley",
        "Rising Son|Takuya Kuroda",
        "Dear Old Stockholm|Stan Getz",
        "Nica's Dream|Horace Silver",
    ]},
    {"name": "Jazz Coolin", "id": "0gikDfPZmk8TdKFFRQjwfP", "tracks": [
        "My Favorite Things|John Coltrane",
        "So What|Miles Davis",
        "Autumn Leaves|Cannonball Adderley",
        "Cantaloupe Island|Herbie Hancock",
        "Chameleon|Herbie Hancock",
        "Fleurette Africaine|Duke Ellington",
        "Tezeta|Mulatu Astatke",
        "Tryin' Times|Roberta Flack",
        "I Put A Spell On You|Nina Simone",
        "All Blues|Miles Davis",
        "Contemplation|McCoy Tyner",
        "Red Clay|Freddie Hubbard",
        "Equinox|John Coltrane",
        "Wise One|John Coltrane",
        "Little Sunflower|Freddie Hubbard",
        "Turiya and Ramakrishna|Alice Coltrane",
        "First Light|Freddie Hubbard",
        "All of Me|Charlie Parker",
        "Blue Train|John Coltrane",
        "Mercy, Mercy, Mercy|Cannonball Adderley",
        "St. Thomas|Sonny Rollins",
        "Strode Rode|Sonny Rollins",
        "The Bridge|Sonny Rollins",
        "Générique|Miles Davis",
        "Chelsea Bridge|Ben Webster",
        "Beatrice|Joe Henderson",
        "Everybody Loves The Sunshine|Takuya Kuroda",
        "Rising Son|Takuya Kuroda",
        "Acknowledgement (A Love Supreme, Pt. I)|John Coltrane",
        "After The Rain|John Coltrane",
        "All The Things You Are|Dizzy Gillespie",
        "You Don't Know What Love Is|Miles Davis",
        "Speak No Evil|Wayne Shorter",
        "Butterfly|Herbie Hancock",
        "Bye Bye Blackbird|Miles Davis",
        "Milestones|Bill Evans",
        "April In Paris|Charlie Parker",
        "Don't Let Me Be Misunderstood|Nina Simone",
        "In Your Own Sweet Way|Wes Montgomery",
        "You Do Something to Me|Sonny Rollins",
        "Maiden Voyage|Herbie Hancock",
        "Autumn Leaves|Ryo Fukui",
        "Theme For Lester Young|Charles Mingus",
        "Gubèlyé|Mulatu Astatke",
        "Resolution|John Coltrane",
        "Every Time I See You|Woody Shaw",
        "Theme for Maxine|Woody Shaw",
        "Stolen Moments|Oliver Nelson",
        "Yearnin'|Oliver Nelson",
        "Tin Tin Deo|Machito",
        "Track A- Solo Dancer|Charles Mingus",
        "Track B- Duet Solo Dancers|Charles Mingus",
        "Solitude|Duke Ellington",
        "Street Fighter Mas|Kamasi Washington",
        "Spain|Chick Corea",
        "Wednesday Night Prayer Meeting|Charles Mingus",
        "Black Narcissus|Joe Henderson",
        "The Golden Flute|Yusef Lateef",
        "Alfie's Theme|Sonny Rollins",
    ]},
]


def _normalize(s):
    s = s.lower()
    s = re.sub(r"\s*[-–—]\s*(remastered|remaster)(\s+\d{4})?\s*$", "", s)
    s = re.sub(r"[‘’.,!?]", "", s)
    s = re.sub(r"\s*&\s*(the\s+)?", " ", s)
    s = re.sub(r"\s*(trio|quartet|quintet|sextet|big band|orchestra)\s*$", "", s)
    s = s.strip()
    return s


def _normalize_key(title, artist):
    return f"{_normalize(title)}|{_normalize(artist)}"


def build_playlists(predictions):
    """Match playlist tracks against predictions and compute aggregate stats."""
    lookup = {}
    for p in predictions:
        key = _normalize_key(p['title'], p['artist'])
        lookup[key] = p

    results = []
    for playlist in PLAYLISTS:
        matched_tracks = []
        all_tracks = []

        for track_str in playlist["tracks"]:
            title, artist = track_str.split("|", 1)
            key = _normalize_key(title.strip(), artist.strip())
            pred = lookup.get(key)

            track_entry = {
                "title": title.strip(),
                "artist": artist.strip(),
                "matched": pred is not None,
            }
            if pred:
                track_entry["actual"] = pred["actual"]
                track_entry["predicted"] = pred["predicted"]
                track_entry["mood_zone"] = pred.get("mood_zone")
                track_entry["era"] = pred.get("era", "Unknown")
                track_entry["energy"] = pred.get("energy", 5)
                track_entry["replayability"] = pred.get("replayability")
                matched_tracks.append(pred)

            all_tracks.append(track_entry)

        # Aggregate stats for matched tracks
        if matched_tracks:
            avg_actual = round(sum(t["actual"] for t in matched_tracks) / len(matched_tracks), 2)
            avg_predicted = round(sum(t["predicted"] for t in matched_tracks) / len(matched_tracks), 2)
            avg_energy = round(sum(t.get("energy", 5) for t in matched_tracks) / len(matched_tracks), 1)
            replay_vals = [t["replayability"] for t in matched_tracks if t.get("replayability") is not None]
            avg_replayability = round(sum(replay_vals) / len(replay_vals), 2) if replay_vals else None

            # Top mood zones across all matched tracks
            zone_counts = {}
            for t in matched_tracks:
                zone = t.get("mood_zone")
                if zone:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
            top_zones = sorted(zone_counts.items(), key=lambda x: -x[1])[:3]
            top_zones = [z[0] for z in top_zones]

            # Top eras
            era_counts = {}
            for t in matched_tracks:
                era = t.get("era", "Unknown")
                era_counts[era] = era_counts.get(era, 0) + 1
            top_eras = sorted(era_counts.items(), key=lambda x: -x[1])[:3]
            top_eras = [e[0] for e in top_eras]
        else:
            avg_actual = None
            avg_predicted = None
            avg_energy = None
            avg_replayability = None
            top_zones = []
            top_eras = []

        results.append({
            "name": playlist["name"],
            "id": playlist["id"],
            "track_count": len(playlist["tracks"]),
            "matched_count": len(matched_tracks),
            "avg_actual": avg_actual,
            "avg_predicted": avg_predicted,
            "avg_energy": avg_energy,
            "avg_replayability": avg_replayability,
            "top_mood_zones": top_zones,
            "top_eras": top_eras,
            "tracks": all_tracks,
        })

    return results


def sync_recco_features(entries):
    """Fetch ReccoBeats audio features for any track with spotify_id but missing them.
    Updates training-data.json in place. Returns number of tracks updated."""
    needs_fetch = [
        (i, e) for i, e in enumerate(entries)
        if e.get("spotify_id")
        and e.get("entity_type") != "album"
        and (e.get("audio_features") or {}).get("acousticness") is None
    ]
    if not needs_fetch:
        return 0

    idx_map = {e["spotify_id"]: i for i, e in needs_fetch}
    spotify_ids = list(idx_map.keys())

    updated = 0
    for batch_start in range(0, len(spotify_ids), 40):
        batch = spotify_ids[batch_start:batch_start + 40]
        url = f"https://api.reccobeats.com/v1/audio-features?ids={','.join(batch)}"
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  ReccoBeats fetch error: {e}")
            continue

        for item in data.get("content", []):
            href = item.get("href", "")
            sid = href.rstrip("/").split("/")[-1] if href else None
            if sid not in idx_map:
                continue
            entry = entries[idx_map[sid]]
            if not entry.get("audio_features"):
                entry["audio_features"] = {}
            entry["audio_features"].update({
                "acousticness":     item.get("acousticness"),
                "danceability":     item.get("danceability"),
                "spotify_energy":   item.get("energy"),
                "instrumentalness": item.get("instrumentalness"),
                "liveness":         item.get("liveness"),
                "loudness":         item.get("loudness"),
                "speechiness":      item.get("speechiness"),
                "spotify_valence":  item.get("valence"),
            })
            updated += 1
        time.sleep(0.3)

    if updated:
        TRAINING_DATA_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    return updated


def main():
    parser = argparse.ArgumentParser(description="Train jazz taste model")
    parser.add_argument("--major", action="store_true",
                        help="Major version bump (new training data era)")
    parser.add_argument("--name", type=str, default="",
                        help="Codename for this version (e.g. Charmander)")
    parser.add_argument("--notes", type=str, default="",
                        help="Version notes (why this release)")
    parser.add_argument("--changelog", type=str, nargs="*", default=None,
                        help="List of changes for this version")
    args = parser.parse_args()

    print("Loading data...")
    all_entries = json.loads(TRAINING_DATA_PATH.read_text())
    print("Syncing ReccoBeats features...")
    synced = sync_recco_features(all_entries)
    if synced:
        print(f"  Fetched ReccoBeats features for {synced} new track(s)")
    tracks, albums = load_data()
    print(f"Loaded {len(tracks)} tracks, {len(albums)} albums")

    print("Engineering features...")
    X, y, feature_names, common_subgenres, common_labels = engineer_features(tracks)

    # Audio features are redundant with era/subgenre/personnel and carry near-zero
    # standalone signal for this taste (audio-only RF R²~0.05; dropping them lifts
    # RF R²). Prune them from the MODEL, but keep a full copy so the Correlations
    # viz panel can still show them (Sound Profile reads audio straight from tracks).
    # Re-run the audio-only / drop-audio bracket at n~300 to see if they wake up.
    AUDIO_FEATURE_KEYS = (
        "acousticness", "danceability", "spotify_energy", "spotify_valence",
        "instrumentalness", "loudness", "speechiness", "liveness", "tempo_bpm",
        "duration", "popularity", "is_live", "is_minor", "is_dorian",
        "missing_energy", "missing_valence", "missing_acous", "missing_instrument",
        "missing_audio", "valence_x_energy",
    )
    audio_cols = [c for c in X.columns if any(k in c for k in AUDIO_FEATURE_KEYS)]
    X_full, feature_names_full = X, feature_names          # kept for viz (correlations)
    X = X.drop(columns=audio_cols)                         # model trains without audio
    feature_names = [f for f in feature_names if f not in audio_cols]
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features "
          f"(pruned {len(audio_cols)} audio-derived; kept for viz)")
    print(f"Common subgenres ({len(common_subgenres)}): {common_subgenres}")
    print(f"Common labels ({len(common_labels)}): {common_labels}")

    print("\nTraining models...")
    model_results = train_models(X, y)
    print(f"Ridge — R²: {model_results['ridge']['r_squared']}, RMSE: {model_results['ridge']['rmse']}")
    print(f"Random Forest — R²: {model_results['random_forest']['r_squared']}, RMSE: {model_results['random_forest']['rmse']}")
    print(f"Best model: {model_results['best_model']}")

    print("\nComputing feature importance...")
    importance = get_feature_importance(model_results, feature_names)
    for f in importance[:5]:
        print(f"  {f['feature']}: {f['importance']} ({f['direction']})")

    print("\nComputing top drivers...")
    top_drivers = get_top_drivers(model_results, feature_names)
    for f in top_drivers[:5]:
        print(f"  {f['feature']}: {f['coefficient']} ({f['direction']})")

    print("\nComputing correlations...")
    # Use the full (unpruned) feature set here so audio features stay in the viz.
    correlations = compute_correlations(X_full, y, feature_names_full)

    print("\nClustering...")
    cluster_results, best_labels, coords = cluster_analysis(
        model_results["X_scaled"], tracks, feature_names
    )
    print(f"Best k: {cluster_results['best_k']}")
    for p in cluster_results["cluster_profiles"]:
        print(f"  Cluster {p['id']}: {p['size']} tracks, mean rating {p['mean_rating']}, mood zones: {p['top_mood_zones']}")

    print("\nBuilding output...")
    predictions = build_predictions(tracks, model_results, best_labels, coords)
    distributions = build_distributions(tracks, best_labels)
    rating_changes = extract_rating_changes(tracks, albums)

    history = load_history()
    best = model_results["best_model"]
    best_metrics = model_results[best] if best == "ridge" else model_results["random_forest"]
    history.append({
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "dataset_size": len(tracks),
        "best_model": best,
        "r_squared": best_metrics["r_squared"],
        "rmse": best_metrics["rmse"],
        "mae": best_metrics["mae"],
        "best_k": cluster_results["best_k"],
        "silhouette_score": max(cluster_results["silhouette_scores"]) if cluster_results["silhouette_scores"] else 0,
        "ridge_r_squared": model_results["ridge"]["r_squared"],
        "ridge_rmse": model_results["ridge"]["rmse"],
        "rf_r_squared": model_results["random_forest"]["r_squared"],
        "rf_rmse": model_results["random_forest"]["rmse"],
    })

    manifest = load_versions()
    new_version = next_version(manifest, is_major=args.major)

    output = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "version": new_version,
            "is_major": args.major,
            "total_tracks": len(tracks),
            "total_albums": len(albums),
            "subgenre_threshold": SUBGENRE_THRESHOLD,
            "best_model": best,
            "feature_count": X.shape[1],
        },
        "models": {
            "ridge": model_results["ridge"],
            "random_forest": model_results["random_forest"],
        },
        "predictions": predictions,
        "feature_importance": importance,
        "top_drivers": top_drivers,
        "correlations": correlations,
        "clusters": cluster_results,
        "distributions": distributions,
        "rating_changes": rating_changes,
        "history": history,
        "playlists": build_playlists(predictions),
        "albums": [{
            "title": a.get("title"),
            "artist": a.get("artist"),
            "year": a.get("year"),
            "label": a.get("label"),
            "era": a.get("era"),
            "rating": a.get("rating"),
            "liked": a.get("liked"),
            "replayability": a.get("replayability"),
            "subgenres": a.get("subgenres", []),
            "notes": a.get("notes"),
            "notable_qualities": a.get("notable_qualities", []),
            "favorite_moments": a.get("favorite_moments"),
        } for a in albums],
    }

    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nExported to {OUTPUT_PATH}")
    print(f"History entries: {len(history)}")

    scaler = model_results["scaler"]
    # Persist the ACTUAL best model (previously this always saved Ridge regardless of
    # best_model — so predict.py silently scored on Ridge, not the winning tree model).
    if best == "ridge":
        best_est = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        best_est.fit(scaler.transform(X), y)
    else:
        best_est = model_results["fitted_model"]   # GBM, already fit on scaler.transform(X)

    eras = sorted(set(t.get("era", "Unknown") for t in tracks))
    joblib.dump({
        "model": best_est,
        "scaler": scaler,
        "feature_names": feature_names,
        "common_subgenres": common_subgenres,
        "common_labels": common_labels,
        "eras": eras,
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} (best={best})")

    metrics_summary = {
        "dataset_size": len(tracks),
        "best_model": best,
        "r_squared": best_metrics["r_squared"],
        "rmse": best_metrics["rmse"],
        "mae": best_metrics["mae"],
        "feature_count": X.shape[1],
        "best_k": cluster_results["best_k"],
        "ridge_r_squared": model_results["ridge"]["r_squared"],
        "ridge_rmse": model_results["ridge"]["rmse"],
        "rf_r_squared": model_results["random_forest"]["r_squared"],
        "rf_rmse": model_results["random_forest"]["rmse"],
    }
    save_version_artifacts(new_version, manifest, metrics_summary,
                           is_major=args.major, name=args.name, notes=args.notes,
                           changelog=args.changelog)
    label = f"{new_version} ({args.name})" if args.name else new_version
    print(f"\nVersion {label} saved")
    if args.major:
        print(f"  Major release — snapshot saved to versions/v{new_version}/")


if __name__ == "__main__":
    main()
