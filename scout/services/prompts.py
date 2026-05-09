"""Use OpenAI to derive an avatar identity from a paper/repo.

We make a single chat-completions call per source that returns a JSON
package:

    - ``image_prompt``     — RunwayML text-to-image prompt for the avatar
    - ``character_name``   — short name shown in the chat UI
    - ``personality``      — system-prompt-style description of how the
                             avatar should converse
    - ``start_script``     — what the avatar says first when it joins a call
    - ``voice_preset``     — one of the curated Runway voice preset ids

The same identity package later feeds both the Runway *image generation*
(for the reference image) and the Runway *avatar creation* call (for the
realtime character). Doing it in one shot keeps the look and the voice
coherent — the LLM sees the whole picture at once.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from openai import OpenAI

log = logging.getLogger("scout.prompts")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
FALLBACK_MODELS = ["gpt-5", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]

# Curated subset of Runway preset voices, chosen to give the LLM a useful
# spread (warm/playful/serious × feminine/masculine/neutral). The full
# catalog is wider; we constrain it so the model picks something we know is
# valid and so we can describe each option clearly.
VOICE_OPTIONS: list[dict[str, str]] = [
    {"id": "victoria", "vibe": "warm, articulate, mid-30s feminine; great for teachers and explainers"},
    {"id": "clara",    "vibe": "bright, friendly, youthful feminine; curious and upbeat"},
    {"id": "luna",     "vibe": "soft, dreamy feminine; storyteller / artistic mood"},
    {"id": "ruby",     "vibe": "confident, sharp feminine; analyst or strategist"},
    {"id": "aurora",   "vibe": "calm, contemplative feminine; researcher / scholar"},
    {"id": "vincent",  "vibe": "calm, low, mature masculine; thoughtful mentor"},
    {"id": "max",      "vibe": "energetic, witty masculine; hacker / builder energy"},
    {"id": "felix",    "vibe": "warm, conversational masculine; friendly engineer"},
    {"id": "marcus",   "vibe": "deep, deliberate masculine; gravitas, philosopher"},
    {"id": "jasper",   "vibe": "playful, theatrical masculine; storyteller"},
    {"id": "morgan",   "vibe": "neutral, smooth, professional; consultant tone"},
    {"id": "sam",      "vibe": "neutral, easygoing; everyman developer"},
]

VOICE_LIST_TEXT = "\n".join(f"- {v['id']}: {v['vibe']}" for v in VOICE_OPTIONS)
VALID_VOICE_IDS = {v["id"] for v in VOICE_OPTIONS}

SYSTEM_PROMPT = f"""You are a casting director and art director for a
small studio that turns software projects and research papers into
talking avatar characters.

Given a single source (a GitHub repo or arXiv paper), invent a character
that visually and vocally embodies that project's vibe, then produce a
JSON package that downstream services will consume verbatim.

You must return ONLY a single JSON object with these keys:

  image_prompt   string   A RunwayML text-to-image prompt for a
                          head-and-shoulders portrait. 2-4 sentences.
                          Single subject (human, animal, robot, mythical
                          creature, or stylised object with a face),
                          facing camera, with a simple blurred backdrop
                          subtly hinting at the project's domain. No
                          text, no logos, no real-person likeness, no
                          trademarked characters.
  character_name string   2-4 words. Memorable, readable. Not the
                          project's literal name; an evocative persona
                          name (e.g. "Tess the Token Tinker").
  personality    string   180-260 words. A second-person system prompt
                          telling the character how to speak and what
                          they care about. Reference the project's
                          actual concepts (mention specific ideas from
                          the README/abstract). Voice should match the
                          chosen voice preset. End with a short list of
                          conversational do's/don'ts.
  start_script   string   1-2 sentences (max 35 words). What the
                          character says first when they join a call.
                          In-character; warm; invites the human to ask
                          something concrete about the project.
  voice_preset   string   Exactly one id from the catalog below. Pick
                          the voice whose vibe most fits the persona
                          you've designed.

Voice catalog:
{VOICE_LIST_TEXT}

Hard rules:
- Output a single JSON object. No prose before or after. No markdown
  fences. No comments inside the JSON.
- Use double quotes for strings. Escape newlines inside strings as \\n.
- voice_preset MUST be one of the listed ids.
- Keep everything safe-for-work and inclusive.
"""

USER_TEMPLATE = """Source type: {source_type}
Title: {title}
{extra}
Short description:
{description}
{readme_block}
Now produce the JSON identity package."""


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _format_user_prompt(source: dict, *, readme: str = "") -> str:
    description = (source.get("description") or "").strip()
    if len(description) > 2400:
        description = description[:2400] + " …"
    extra_lines: list[str] = []
    meta = source.get("meta") or {}
    if source.get("source") == "github":
        if meta.get("language"):
            extra_lines.append(f"Primary language: {meta['language']}")
        if meta.get("topics"):
            extra_lines.append(f"Topics: {', '.join(meta['topics'][:8])}")
        if meta.get("stars") is not None:
            extra_lines.append(f"Stars: {meta['stars']}")
    elif source.get("source") == "arxiv":
        if meta.get("primary_category"):
            extra_lines.append(f"Primary category: {meta['primary_category']}")
        if source.get("subtitle"):
            extra_lines.append(f"Authors: {source['subtitle']}")

    readme_block = ""
    if readme.strip():
        readme_block = f"\nREADME (truncated to a few thousand chars):\n{readme.strip()}\n"

    return USER_TEMPLATE.format(
        source_type=source.get("source", "project"),
        title=source.get("title", "Untitled"),
        extra="\n".join(extra_lines),
        description=description or "(no description provided)",
        readme_block=readme_block,
    )


# Some models wrap JSON in ```json fences even when told not to. Strip if so.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_identity(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    # Pull out the first {...} block defensively.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in model output: {raw[:200]}")
    obj = json.loads(text[start : end + 1])
    required = {"image_prompt", "character_name", "personality", "start_script", "voice_preset"}
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"identity JSON missing keys: {sorted(missing)}")
    if obj["voice_preset"] not in VALID_VOICE_IDS:
        # Fail soft: pick a sensible default.
        log.warning(
            "voice_preset %r not in catalog, defaulting to victoria",
            obj["voice_preset"],
        )
        obj["voice_preset"] = "victoria"
    for key in ("image_prompt", "character_name", "personality", "start_script"):
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"identity field {key!r} is empty")
    return obj


def generate_identity(
    source: dict,
    *,
    readme: str = "",
    model: Optional[str] = None,
) -> dict:
    """Return a complete avatar identity package for ``source``.

    Pass ``readme`` (raw markdown / text, already truncated to a sensible
    size) when available — for GitHub repos it makes the persona vastly
    sharper.
    """
    client = _client()
    user_msg = _format_user_prompt(source, readme=readme)
    candidates = [model or DEFAULT_MODEL] + [
        m for m in FALLBACK_MODELS if m != (model or DEFAULT_MODEL)
    ]
    last_err: Optional[Exception] = None
    for m in candidates:
        try:
            kwargs: dict = {
                "model": m,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            }
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
            except Exception:
                # Older models may not support response_format; retry without.
                resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                last_err = RuntimeError(f"Model {m} returned empty content")
                continue
            return _parse_identity(text)
        except Exception as e:
            log.warning("identity model %s failed: %s", m, e)
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to generate identity with any of {candidates}: {last_err}"
    )


def generate_prompt(source: dict, *, readme: str = "", model: Optional[str] = None) -> str:
    """Backwards-compatible thin wrapper that returns just the image prompt."""
    return generate_identity(source, readme=readme, model=model)["image_prompt"]


def describe_failure(err: Exception) -> str:
    return json.dumps({"type": type(err).__name__, "message": str(err)})
