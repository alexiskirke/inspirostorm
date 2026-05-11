"""Scout: scan GitHub or arXiv, pick up to 3 items, generate Runway avatars."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load env BEFORE importing services that read env at module scope.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

from .services import (
    arxiv_scan,
    avatars,
    brainstorm,
    github_scan,
    images,
    knowledge,
    movie_pipeline,
    prompts,
    storage,
    summariser,
    synthesis as synthesis_svc,
    uploads,
    web_scrape,
    youtube_transcript,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("scout")

app = FastAPI(title="Scout: Papers & Repos to Avatars")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

storage.init_db()
log.info("DATA_DIR = %s", storage.DATA_DIR)

EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="img")
AVATAR_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="avatar")
KB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kb")
SUMMARISER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sum")
MOVIE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="movie")


# ----- request/response models ------------------------------------------------


class GithubScanRequest(BaseModel):
    username: str
    language: Optional[str] = None
    days: Optional[int] = Field(default=None, ge=1, le=3650)
    limit: int = Field(default=30, ge=1, le=100)


class ArxivScanRequest(BaseModel):
    query: str = ""
    categories: Optional[list[str]] = None
    days: Optional[int] = Field(default=None, ge=1, le=3650)
    limit: int = Field(default=30, ge=1, le=100)


class GenerateRequest(BaseModel):
    items: list[dict] = Field(min_length=1, max_length=3)
    model: Optional[str] = None
    ratio: Optional[str] = None


class UrlScanRequest(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None


class YoutubeScanRequest(BaseModel):
    url: str
    title: Optional[str] = None
    description: Optional[str] = None


# ----- routes -----------------------------------------------------------------


CHAT_BASE_URL = os.environ.get("CHAT_BASE_URL", "http://localhost:5173")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "default_image_model": images.DEFAULT_MODEL,
            "default_ratio": images.DEFAULT_RATIO,
            "chat_base_url": CHAT_BASE_URL,
        },
    )


@app.get("/brainstorm", response_class=HTMLResponse)
def brainstorm_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "brainstorm.html",
        {
            "chat_base_url": CHAT_BASE_URL,
            # Pass only the bool — never leak the actual room URL (which
            # contains the embedded pwd) to every browser that hits /brainstorm.
            "personal_zoom_active": bool(brainstorm.PERSONAL_ZOOM_ROOM),
        },
    )


@app.post("/api/scan/github")
def scan_github(req: GithubScanRequest) -> dict:
    try:
        results = github_scan.fetch_repos(
            req.username,
            language=req.language,
            days=req.days,
            limit=req.limit,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("github scan failed")
        raise HTTPException(status_code=502, detail=str(e))
    return {"items": results}


@app.post("/api/scan/upload")
async def scan_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
) -> dict:
    """Persist a custom-source upload (.zip or .pdf) and return a single
    'item' in the same shape as github/arxiv scans, ready to be selected
    in the gallery and turned into an avatar."""
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not read upload: {e}")
    try:
        stored = uploads.save_upload(
            title=title,
            description=description,
            filename=file.filename or "upload",
            content=content,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("upload failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": [stored.source]}


@app.post("/api/scan/url")
def scan_url(req: UrlScanRequest) -> dict:
    """Fetch a webpage, strip markup, ask GPT to keep only the relevant
    body text, and return a single Scout source (same shape as
    github/arxiv/upload scans). The cleaned text is persisted so it can
    be re-used as the avatar's knowledge base."""
    try:
        stored = web_scrape.fetch_and_clean(
            url=req.url,
            title=req.title,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        log.exception("url scrape failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": [stored.source]}


@app.post("/api/scan/youtube")
def scan_youtube(req: YoutubeScanRequest) -> dict:
    """Pull a YouTube video's transcript and return a single Scout
    source. No GPT cleaning — captions are already plain spoken text,
    we just strip [Music]/[Applause] markers and join into paragraphs.
    English-first language fallback with translation as last resort."""
    try:
        stored = youtube_transcript.fetch_and_store(
            url=req.url,
            title=req.title,
            description=req.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        log.exception("youtube transcript fetch failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"items": [stored.source]}


@app.post("/api/scan/arxiv")
def scan_arxiv(req: ArxivScanRequest) -> dict:
    try:
        results = arxiv_scan.fetch_papers(
            req.query,
            categories=req.categories,
            days=req.days,
            limit=req.limit,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("arxiv scan failed")
        raise HTTPException(status_code=502, detail=str(e))
    return {"items": results}


def _process_generation(gen_id: str, prompt: str, model: str, ratio: str) -> None:
    images.generate_and_store(
        gen_id, prompt=prompt, model=model, ratio=ratio
    )


def _readme_for(source: dict) -> str:
    """Return rich textual context for the identity-generation LLM call.

    - GitHub: fetched README via the GitHub API.
    - arXiv: nothing extra (the abstract is already in ``description``).
    - Upload: the ``preview_text`` we precomputed at upload time
      (README from the zip, or the first ~6k chars of the PDF).
    - URL: the ``preview_text`` we precomputed at scrape time
      (GPT-cleaned main-body text from the page).
    - YouTube: the ``preview_text`` we precomputed at ingest time
      (joined transcript snippets, ``[Music]`` markers stripped).
    """
    src = source.get("source")
    if src in ("upload", "url", "youtube"):
        return source.get("preview_text") or ""
    if src != "github":
        return ""
    full_name = (source.get("subtitle") or "").strip()
    if not full_name:
        url = source.get("url") or ""
        if url.startswith("https://github.com/"):
            full_name = url.removeprefix("https://github.com/").strip("/")
    if not full_name:
        return ""
    try:
        return github_scan.fetch_readme(full_name)
    except Exception:
        log.exception("readme fetch failed for %s", full_name)
        return ""


@app.post("/api/generate")
def generate(req: GenerateRequest, _bg: BackgroundTasks) -> dict:
    """Start image generation for each selected item.

    For each source we (1) fetch enrichment context (README for repos),
    (2) ask GPT for a full identity package (image prompt + character
    name + personality + start script + voice preset), (3) launch a
    Runway image generation in a background thread. The frontend polls
    /api/generations for completion.
    """
    chosen_model = req.model or images.DEFAULT_MODEL
    chosen_ratio = req.ratio or images.DEFAULT_RATIO
    created: list[dict] = []
    # Track voices picked so far in this batch so the 2nd / 3rd identity
    # call can't reuse them. Filtered catalog is passed into GPT via the
    # user-message voice list — see prompts._format_user_prompt.
    chosen_voices_this_batch: set[str] = set()
    for source in req.items:
        if not source.get("id") or not source.get("source"):
            raise HTTPException(
                status_code=400, detail="each item must have id and source"
            )
        readme = _readme_for(source)
        if readme:
            log.info(
                "fetched readme for %s (%d chars)", source["id"], len(readme)
            )
        try:
            identity = prompts.generate_identity(
                source,
                readme=readme,
                excluded_voice_ids=chosen_voices_this_batch,
            )
        except Exception as e:
            log.exception("identity generation failed for %s", source.get("id"))
            raise HTTPException(
                status_code=502, detail=f"identity generation failed: {e}"
            )
        chosen_voices_this_batch.add(identity["voice_preset"])
        gen_id = storage.create_generation(
            source=source,
            prompt=identity["image_prompt"],
            model=chosen_model,
            ratio=chosen_ratio,
            identity=identity,
            readme_excerpt=readme,
        )
        EXECUTOR.submit(
            _process_generation,
            gen_id,
            identity["image_prompt"],
            chosen_model,
            chosen_ratio,
        )
        created.append(
            {
                "id": gen_id,
                "source_id": source["id"],
                "prompt": identity["image_prompt"],
                "character_name": identity["character_name"],
                "voice_preset": identity["voice_preset"],
            }
        )
    return {"generations": created}


# ----- avatar (custom Runway character) ---------------------------------------


@app.post("/api/generations/{gen_id}/avatar")
def create_avatar(gen_id: str) -> dict:
    """Kick off Runway custom-avatar creation in the background."""
    rec = storage.get_generation(gen_id)
    if not rec:
        raise HTTPException(status_code=404, detail="generation not found")
    if rec.get("status") != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"image is not ready (status={rec.get('status')})",
        )
    if rec.get("runway_avatar_id"):
        return {
            "status": "ready",
            "runway_avatar_id": rec["runway_avatar_id"],
            "voice_preset": rec.get("voice_preset"),
            "character_name": rec.get("character_name"),
        }
    if rec.get("avatar_status") == "creating":
        return {"status": "creating"}
    AVATAR_EXECUTOR.submit(avatars.create_avatar_for_generation, gen_id)
    storage.update_generation(gen_id, avatar_status="creating", avatar_error=None)
    return {"status": "creating"}


# -----------------------------------------------------------------------------
# Brainstorm: pair two custom avatars, dispatch to a meeting, persist memory.
# (Phase 3 of the brainstorm roadmap; UI lives in scout/templates/brainstorm.html.)
# -----------------------------------------------------------------------------


class BrainstormStartBody(BaseModel):
    avatar_a_gen_id: str
    avatar_b_gen_id: str
    meeting_url: Optional[str] = None
    topic: Optional[str] = None


@app.post("/api/brainstorm/start")
def brainstorm_start(body: BrainstormStartBody) -> dict:
    """Find-or-create the thread for this pair, then dispatch a session."""
    try:
        thread = brainstorm.find_or_create_thread(
            body.avatar_a_gen_id,
            body.avatar_b_gen_id,
            topic_seed=body.topic,
        )
        sess = brainstorm.start_session(
            thread["id"],
            meeting_url=body.meeting_url,
            topic=body.topic,
        )
    except (LookupError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception("brainstorm start failed")
        raise HTTPException(status_code=502, detail=str(e))
    return {"thread": thread, "session": sess}


@app.post("/api/brainstorm/sessions/{session_id}/end")
def brainstorm_end(session_id: str, reason: str = "manual") -> dict:
    try:
        sess = brainstorm.end_session(session_id, reason=reason)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("brainstorm end failed")
        raise HTTPException(status_code=500, detail=str(e))
    # Fire summariser in background so the HTTP request returns
    # promptly. Failures are swallowed by summarise_session_safe.
    SUMMARISER_EXECUTOR.submit(summariser.summarise_session_safe, session_id)
    return sess


@app.post("/api/brainstorm/sessions/{session_id}/summarise")
def brainstorm_summarise(session_id: str) -> dict:
    """Manually re-run the gpt-5.1 summariser on an already-ended session."""
    try:
        return summariser.summarise_session(session_id) or {}
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("brainstorm summarise failed")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/brainstorm/threads")
def brainstorm_threads() -> dict:
    return {"items": brainstorm.list_threads()}


@app.post("/api/brainstorm/reset")
def brainstorm_reset() -> dict:
    """Wipe ALL brainstorm history (threads, sessions, rolling memory,
    syntheses + per-synthesis movie/composite files). Avatars + their
    Runway custom characters are preserved."""
    counts = storage.reset_brainstorm()
    log.info(
        "brainstorm reset: threads=%d sessions=%d state=%d syntheses=%d "
        "movies=%d composites=%d",
        counts["threads"], counts["sessions"], counts["state_rows"],
        counts["syntheses"], counts["movies_deleted"], counts["composites_deleted"],
    )
    return {"reset": True, "counts": counts}


@app.get("/api/brainstorm/threads/{thread_id}")
def brainstorm_thread(thread_id: str) -> dict:
    th = brainstorm.get_thread(thread_id)
    if not th:
        raise HTTPException(status_code=404, detail="thread not found")
    return {
        "thread": th,
        "state": brainstorm.get_thread_state(thread_id),
        "sessions": brainstorm.list_sessions(thread_id),
    }


@app.get("/api/brainstorm/sessions/{session_id}")
def brainstorm_session(session_id: str) -> dict:
    s = brainstorm.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@app.post("/api/brainstorm/threads/{thread_id}/synthesise")
def brainstorm_synthesise(thread_id: str) -> dict:
    """Run gpt-5.4 synthesis across the whole thread. Returns the new
    synthesis row (text_md, movie_pitch, ideas)."""
    try:
        return synthesis_svc.synthesise_thread(thread_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/brainstorm/threads/{thread_id}/synthesis")
def brainstorm_synthesis_list(thread_id: str) -> dict:
    return {"items": synthesis_svc.list_synthesis(thread_id)}


@app.post("/api/brainstorm/synthesis/{synth_id}/movie")
def brainstorm_synth_movie(synth_id: str) -> dict:
    """Kick off the 3-clip movie pipeline for a synthesis row in the
    background. UI polls /api/brainstorm/threads/<thread_id>/synthesis
    until movie_status flips to 'ready' (or 'failed') and a movie_path
    appears."""
    rec = storage.get_generation  # just confirms storage import is live
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT thread_id, movie_status FROM brainstorm_synthesis WHERE id = ?",
            (synth_id,),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="synthesis not found")
    if row["movie_status"] == "building":
        return {"status": "building"}
    storage._connect().close()
    with storage._LOCK, storage._connect() as conn:
        conn.execute(
            "UPDATE brainstorm_synthesis SET movie_status='building', movie_error=NULL WHERE id = ?",
            (synth_id,),
        )
    MOVIE_EXECUTOR.submit(movie_pipeline.make_movie_for_synthesis_safe, synth_id)
    return {"status": "building"}


@app.get("/data/movies/{filename}")
def serve_movie(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = storage.MOVIES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="movie not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/generations/{gen_id}/knowledge")
def attach_knowledge(gen_id: str) -> dict:
    """(Re)build the avatar's knowledge base from the source.

    Useful for avatars that were created before the knowledge feature
    existed, or to refresh the docs after the upstream README/paper
    changes. Replaces all currently attached documents.
    """
    rec = storage.get_generation(gen_id)
    if not rec:
        raise HTTPException(status_code=404, detail="generation not found")
    if not rec.get("runway_avatar_id"):
        raise HTTPException(
            status_code=409, detail="avatar has not been created yet"
        )
    if rec.get("kb_status") == "building":
        return {"status": "building"}
    KB_EXECUTOR.submit(knowledge.attach_knowledge_for_generation, gen_id)
    storage.update_generation(gen_id, kb_status="building", kb_error=None)
    return {"status": "building"}


@app.get("/api/generations")
def list_generations(ids: Optional[str] = None, limit: int = 100) -> dict:
    if ids:
        id_list = [s for s in ids.split(",") if s]
        return {"items": storage.get_generations(id_list)}
    return {"items": storage.list_generations(limit=limit)}


@app.get("/api/generations/{gen_id}")
def get_generation(gen_id: str) -> dict:
    rec = storage.get_generation(gen_id)
    if not rec:
        raise HTTPException(status_code=404, detail="generation not found")
    return rec


@app.delete("/api/generations/{gen_id}")
def delete_generation(gen_id: str) -> dict:
    """Tear down a generated avatar.

    Steps (in order):
      1. Refuse if any brainstorm session involving this avatar is
         currently ``status='live'`` — kill the session first.
      2. Best-effort Runway cleanup: delete the custom character and
         every attached document. Failures here are reported but do
         not block local cleanup (the user explicitly asked to delete).
      3. Delete the local DB row + on-disk image.

    Brainstorm threads / sessions / syntheses that reference this
    avatar are LEFT INTACT for history. They'll point at a now-missing
    generation row, and the brainstorm UI shows that as a deleted
    participant.
    """
    rec = storage.get_generation(gen_id)
    if not rec:
        raise HTTPException(status_code=404, detail="generation not found")

    live = storage.live_brainstorm_sessions_for_avatar(gen_id)
    if live:
        raise HTTPException(
            status_code=409,
            detail=(
                f"avatar is currently in a live brainstorm session "
                f"({', '.join(live)}). End the session before deleting."
            ),
        )

    runway_result = avatars.delete_runway_artifacts(gen_id)
    deleted = storage.delete_generation(gen_id)
    if not deleted:
        # Should not happen — we just read the row — but be defensive.
        raise HTTPException(status_code=404, detail="generation vanished mid-delete")

    log.info(
        "deleted gen=%s runway_avatar=%s runway_docs=%d/%d errors=%s",
        gen_id,
        runway_result.get("avatar_deleted"),
        runway_result.get("documents_deleted", 0),
        runway_result.get("documents_total", 0),
        runway_result.get("errors") or [],
    )
    return {
        "deleted": True,
        "gen_id": gen_id,
        "runway": runway_result,
    }


@app.get("/api/generations/by-source/{source_id:path}")
def by_source(source_id: str) -> dict:
    return {"items": storage.list_generations_for_source(source_id)}


@app.get("/data/images/{filename}")
def serve_image(filename: str) -> FileResponse:
    """Serve generated images from the persistent data dir."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = storage.IMAGES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(path)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    """Serve the .ico directly so browser auto-requests for /favicon.ico
    don't 404 (most browsers fetch this even when <link rel='icon'> is set)."""
    return FileResponse(
        ROOT / "static" / "favicon" / "favicon.ico",
        media_type="image/x-icon",
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "data_dir": str(storage.DATA_DIR),
        "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "runway_key_present": bool(
            os.environ.get("RUNWAYML_API_KEY")
            or os.environ.get("RUNWAYML_API_SECRET")
        ),
    }
