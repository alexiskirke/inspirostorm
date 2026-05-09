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

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

from .services import arxiv_scan, avatars, github_scan, images, knowledge, prompts, storage

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
    """Pull a README (or equivalent rich context) for ``source``.

    For GitHub: GET /repos/{full_name}/readme. For arXiv: nothing extra
    needed — the abstract is already the canonical summary and we pass
    it through as ``description``.
    """
    if source.get("source") != "github":
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
            identity = prompts.generate_identity(source, readme=readme)
        except Exception as e:
            log.exception("identity generation failed for %s", source.get("id"))
            raise HTTPException(
                status_code=502, detail=f"identity generation failed: {e}"
            )
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
