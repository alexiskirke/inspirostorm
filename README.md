# inspirostorm

Pair Runway avatars and let them brainstorm with each other inside a real
video meeting. Scout reads papers and repos, mints a custom Runway avatar
for each idea, then sends two of those avatars into a Zoom/Meet/Teams as
separate participants. They talk, the transcripts are summarised, and on
demand a deeper synthesis is distilled into an idea + movie pitch.

Runway hackathon submission.

## What's in the box

| Dir | Stack | Port | What it does |
|---|---|---|---|
| [scout/](scout/) | Python · FastAPI | 8000 | Scans arXiv / GitHub, generates a Runway avatar per item, orchestrates brainstorms, runs the synthesis + movie pipeline. Brainstorm UI at `/brainstorm`. |
| [meet/](meet/) | Node · Express | 3000 | Sends a Runway character into a Zoom/Meet/Teams call as a participant (via Recall.ai). Forked from [runwayml/runway-characters-meet](https://github.com/runwayml/runway-characters-meet) with custom multi-bot + personality-override support. |
| [app/](app/) | React · Vite · Express | 5173 / 3001 | Standalone avatar chat — the "Chat live →" destination for a single avatar. |
| [tier2-test/](tier2-test/) | React · Vite · Express | 5174 / 3002 | Testbench for running two preset avatars concurrently. |

## How a brainstorm flows

```
  ┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐
  │ arXiv / GitHub  │ ──►  │ scout (FastAPI) │ ──►  │ custom avatars   │
  └─────────────────┘      │  brainstorm UI  │      │ (Runway image →  │
                           └────────┬────────┘      │  Runway avatar)  │
                                    │               └────────┬─────────┘
                                    │ pick 2 avatars +       │
                                    │ a meeting URL          ▼
                                    │                ┌──────────────────┐
                                    └──────────────► │ meet (Recall.ai) │
                                                     │ 2 bots → Zoom    │
                                                     └────────┬─────────┘
                                                              │ transcripts
                                                              ▼
                                                     ┌──────────────────┐
                                                     │ scout summariser │
                                                     │ (rolling memory) │
                                                     └────────┬─────────┘
                                                              │ on demand
                                                              ▼
                                                     ┌──────────────────┐
                                                     │ scout synthesis  │
                                                     │ idea + movie     │
                                                     └──────────────────┘
```

## Prerequisites

- Python 3.11+ and Node.js 18+
- A [Runway API key](https://dev.runwayml.com)
- An [OpenAI API key](https://platform.openai.com)
- A [Recall.ai API key](https://www.recall.ai/) (only needed for the brainstorm + meet pieces)
- `cloudflared` available on PATH (the tunnel watchdog uses `npx cloudflared`)

## Setup

```sh
# Python deps for scout
python3 -m venv venv
venv/bin/pip install -r scout/requirements.txt

# Node deps for the three Node subprojects
( cd meet && npm install )
( cd app  && npm install )
( cd tier2-test && npm install )   # optional — only for tier-2 concurrency tests
```

Create a top-level `.env`:

```sh
RUNWAYML_API_KEY=...
OPENAI_API_KEY=...

# Optional: pin the brainstorm to your own Zoom Personal Meeting Room
PERSONAL_ZOOM_ROOM=https://us05web.zoom.us/j/...

# Optional: raise GitHub scan rate-limit (60/hr → 5000/hr)
GITHUB_TOKEN=...
```

And a `meet/.env` (copy from [meet/.env.example](meet/.env.example)):

```sh
RECALL_API_KEY=...
RECALL_REGION=us-west-2
# PUBLIC_URL is set by the tunnel watchdog at runtime — leave blank locally
```

## Run

```sh
./run                # scout + tunnel watchdog (meet + cloudflared) + app
./run --no-app       # skip the React avatar chat
./run --scout-only   # just FastAPI (brainstorm features won't work)
./stop               # graceful teardown of every process this stack spawns
```

Then open:

- Scout gallery: <http://localhost:8000/>
- Brainstorm UI: <http://localhost:8000/brainstorm>
- Avatar chat:   <http://localhost:5173/>

## Tier-2 concurrency test

[tier2-test/](tier2-test/) is a standalone testbench that confirms two
preset Runway avatars can run in parallel under the tier-2 plan. It's
isolated on its own ports so it doesn't fight with the main stack:

```sh
cd tier2-test && npm run dev
# open http://localhost:5174
```

## Credits

The `meet/` subdirectory is a fork of
[runwayml/runway-characters-meet](https://github.com/runwayml/runway-characters-meet)
with multi-bot + personality-override extensions. Original repo MIT-licensed.
