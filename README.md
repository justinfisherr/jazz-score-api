# Jazz Score API

Serverless Spotify search + taste-model scoring, for the claude-metrics dashboard.

- `GET /api/score?q=moanin%20art%20blakey`
- `GET /api/score?title=So%20What&artist=Miles%20Davis`

The model files (`model.joblib`, `training-data.json`) are copies from `~/claude-metrics`.
After retraining there, refresh them:

    cp ~/claude-metrics/model.joblib ~/claude-metrics/training-data.json ~/Projects/jazz-score-api/

Requires `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` env vars (set on Vercel).

## AI recommendations

`POST /api/recommend` (requires `x-app-password`) asks OpenRouter for jazz
recommendations grounded in the current Postgres ratings, then resolves and
scores tracks with the live model.

Env:
- `OPENROUTER_API_KEY` — OpenRouter key (stored in Vercel; local copy at `~/.openrouter_key`)
- `OPENROUTER_MODEL` — optional, defaults to `deepseek/deepseek-v4.1-flash`
