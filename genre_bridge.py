"""Genre bridge: map non-jazz genres onto jazz vocabulary so the taste model can
score them without touching train.py or model.joblib.

The jazz model only understands jazz words (subgenres, eras, instruments,
moods). A raw non-jazz track collapses to the "unseen genre" prior (~4.5).
`apply()` translates a track's genre into the nearest jazz idiom so the score
is a real read of the taste profile rather than "not jazz".

Read-only: nothing here mutates the dataset or the model.
"""

GENRES = {
    "neo-soul": {
        "subgenres": ["soul jazz", "vocal jazz", "jazz funk"],
        "era": "Contemporary",
        "primary_instrument": "vocals",
        "instrumentation": ["vocals", "drums", "bass", "electric piano", "keyboards"],
        "moods": ["smooth", "sensual", "warm", "groovy", "intimate"],
    },
    "soul": {
        "subgenres": ["soul jazz", "vocal jazz", "blues"],
        "era": "Soul/Funk",
        "primary_instrument": "vocals",
        "instrumentation": ["vocals", "drums", "bass", "guitar", "keyboards"],
        "moods": ["passionate", "warm", "groovy", "joyful"],
    },
    "r&b": {
        "subgenres": ["soul jazz", "vocal jazz", "jazz funk"],
        "era": "Contemporary",
        "primary_instrument": "vocals",
        "instrumentation": ["vocals", "drums", "bass", "keyboards"],
        "moods": ["smooth", "sexy", "intimate", "nostalgic"],
    },
    "funk": {
        "subgenres": ["jazz funk", "jazz-funk"],
        "era": "Soul/Funk",
        "primary_instrument": "bass",
        "instrumentation": ["bass", "drums", "guitar", "keyboards", "electric piano"],
        "moods": ["groovy", "fun", "energetic", "swinging"],
    },
    "hip-hop": {
        "subgenres": ["jazz funk", "modern jazz"],
        "era": "Contemporary",
        "primary_instrument": "drums",
        "instrumentation": ["drums", "bass", "keyboards", "vocals"],
        "moods": ["groovy", "cool", "assertive", "dynamic"],
    },
    "electronic": {
        "subgenres": ["modern jazz", "jazz fusion"],
        "era": "Electric",
        "primary_instrument": "keyboards",
        "instrumentation": ["keyboards", "drums", "bass"],
        "moods": ["experimental", "atmospheric", "spacious", "mysterious"],
    },
    "rock": {
        "subgenres": ["jazz fusion", "modern jazz"],
        "era": "Electric",
        "primary_instrument": "guitar",
        "instrumentation": ["guitar", "bass", "drums", "keyboards"],
        "moods": ["energetic", "intense", "raw", "dramatic"],
    },
    "blues": {
        "subgenres": ["blues", "soul jazz"],
        "era": "Swing",
        "primary_instrument": "guitar",
        "instrumentation": ["guitar", "bass", "drums", "vocals"],
        "moods": ["bluesy", "raw", "mournful", "passionate"],
    },
    "classical": {
        "subgenres": ["orchestral jazz", "ballad"],
        "era": "Contemporary",
        "primary_instrument": "piano",
        "instrumentation": ["strings", "piano"],
        "moods": ["serene", "lush", "beautiful", "tender"],
    },
    "folk": {
        "subgenres": ["standard", "ballad"],
        "era": "Contemporary",
        "primary_instrument": "guitar",
        "instrumentation": ["guitar", "vocals", "bass"],
        "moods": ["tender", "intimate", "gentle", "melancholic"],
    },
}


def genres():
    return sorted(GENRES)


def apply(candidate, genre):
    """Return a copy of `candidate` with the genre's jazz mapping applied.

    Unknown/empty genre returns the candidate unchanged.
    """
    if not genre:
        return candidate
    mapping = GENRES.get((genre or "").strip().lower())
    if not mapping:
        return candidate
    out = dict(candidate)
    for key, value in mapping.items():
        out[key] = list(value) if isinstance(value, list) else value
    return out
