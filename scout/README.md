# Scout — papers/repos → Runway avatars

Scan GitHub or arXiv, pick up to 3 items, and let GPT + Runway draw a
head-and-shoulders avatar that captures each one's vibe. Generated images and
their source-item link are persisted under `DATA_DIR` (SQLite + files).

## Local run

```bash
# Uses the venv at ../venv. Make sure .env (in this folder OR the parent)
# has RUNWAYML_API_KEY and OPENAI_API_KEY.
../venv/bin/uvicorn scout.main:app --reload --port 8000
# Open http://localhost:8000
```

Optional env vars:

- `DATA_DIR` — where to keep the SQLite db and generated images. Defaults to
  `./scout/data` locally. On Railway, point this at the mounted volume
  (e.g. `/data`).
- `OPENAI_MODEL` — defaults to `gpt-5.2`, with automatic fallback to
  `gpt-5`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini` if the chosen model isn't
  available.
- `RUNWAY_IMAGE_MODEL` — defaults to `gen4_image` (also supports
  `gen4_image_turbo` and `gemini_2.5_flash`).
- `RUNWAY_IMAGE_RATIO` — defaults to `1024:1024`.
- `GITHUB_TOKEN` — optional, raises GitHub's rate limit from 60/hr to 5000/hr.

## Railway notes (later)

When deploying, mount a persistent volume (e.g. at `/data`) and set
`DATA_DIR=/data`. Everything (SQLite db + image files) lives there, so
restarts/redeploys won't lose history.
