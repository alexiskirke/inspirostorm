"""Phase-7 dress rehearsal: produce a single-clip "movie" with separate
voice and music tracks, mixed in ffmpeg.

Pipeline (all in one script for hackathon iteration speed; will move to
``services/`` once the full brainstorm pipeline wires it in):

    1. Ask gpt-5.1 to invent a SHORT, POSITIVE, BACKGROUND music vibe
       grounded in the topic + dialogue. Output: a one-sentence prompt
       suitable for ``sound_effect`` (eleven_text_to_sound_v2) of about
       30 words, biased toward instrumentation+tempo+mood and tagged
       "background only".
    2. Call Runway ``image_to_video`` (veo3.1_fast, audio=True) with the
       composite still as the first frame and a dialogue-only prompt
       that explicitly tells Veo NOT to add music ("just speech + quiet
       ambient room tone"). This gives a clean dialogue track.
    3. Call Runway ``sound_effect`` for ``duration`` seconds of music
       using gpt-5.1's prompt.
    4. Use ffmpeg to mix: full-volume Veo audio + music at the user-
       configurable music_db gain (default -18 dB ≈ 12% so dialogue
       always wins), output a new MP4 keeping the original video.

Run:
  venv/bin/python -m scout.scripts.test_movie_with_music \
      --composite scout/data/composites/<id>.png \
      --topic "the shared craft of building learning machines from first principles" \
      --speech "the chrome robot says X, the scholar replies Y" \
      --duration 8

Optional: --music-db -15, --music-model eleven_text_to_sound_v2,
          --output some/path.mp4, --keep-intermediate
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "scout" / ".env")

from openai import OpenAI                             # noqa: E402
from runwayml import RunwayML                         # noqa: E402

from scout.services import movies, storage            # noqa: E402

log = logging.getLogger("scout.test_movie_with_music")


# ---------------------------------------------------------------------------
# Step 1: gpt-5.1 invents a background music vibe
# ---------------------------------------------------------------------------


VIBE_SYSTEM = """You are a film music director.

You will be given:
  - a one-sentence TOPIC for a short cinematic clip
  - the on-screen DIALOGUE between two characters

Your job: invent a short BACKGROUND music vibe that supports the topic
and dialogue without distracting from the speech. The music must be:
  - positive in mood (warm, hopeful, inquisitive — never tense, dark
    or sad unless the topic clearly demands it)
  - simple and minimal — clearly UNDERSCORE, not a song. Think one or
    two instruments at most, gentle tempo, no strong melody hooks.
  - clearly intended as background — should NEVER make a listener stop
    paying attention to the dialogue
  - evocative of the topic at a high level, but not literal

Return ONLY a single JSON object:

  vibe         string  1-2 sentences in plain English describing the
                       intended music style for a creative brief.
  sfx_prompt   string  ~30-40 words MAX. The exact prompt that will be
                       sent to ElevenLabs eleven_text_to_sound_v2 to
                       generate the audio. Focus on instrumentation,
                       tempo, mood adjectives, and the explicit framing
                       'soft background underscore, low volume, gentle,
                       no melody hook, no vocals'. Begin with the most
                       important descriptors.

Hard rules:
- Output a single JSON object, no prose around it, no markdown fences.
- sfx_prompt MUST end with the phrase 'soft background underscore, no
  vocals' so the audio model knows to keep it minimal.
- Keep everything safe-for-work.
"""


def invent_vibe(topic: str, speech: str, model: str = "gpt-5.1") -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    user = (
        f"TOPIC:\n{topic.strip()}\n\n"
        f"DIALOGUE:\n{speech.strip()}\n\n"
        "Now produce the JSON object."
    )
    log.info("asking %s for music vibe…", model)
    fallbacks = [model, "gpt-5", "gpt-4.1", "gpt-4o"]
    last_err: Optional[Exception] = None
    for m in fallbacks:
        try:
            kwargs: dict = {
                "model": m,
                "messages": [
                    {"role": "system", "content": VIBE_SYSTEM},
                    {"role": "user", "content": user},
                ],
            }
            try:
                resp = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
            except Exception:
                resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            obj = json.loads(text[text.find("{") : text.rfind("}") + 1])
            if not obj.get("vibe") or not obj.get("sfx_prompt"):
                raise ValueError("vibe/sfx_prompt missing in model output")
            obj["model_used"] = m
            return obj
        except Exception as e:
            log.warning("vibe model %s failed: %s", m, e)
            last_err = e
    raise RuntimeError(f"vibe-invent failed across {fallbacks}: {last_err}")


# ---------------------------------------------------------------------------
# Step 3: sound_effect music generator
# ---------------------------------------------------------------------------


def generate_music(
    sfx_prompt: str,
    duration_s: float,
    output_path: Path,
    *,
    model: str = "eleven_text_to_sound_v2",
    poll_timeout_s: int = 180,
) -> dict:
    """Call Runway sound_effect, poll, download to ``output_path``.
    Returns a dict with task_id, output_url, bytes_written."""
    api_key = os.environ.get("RUNWAYML_API_KEY") or os.environ.get(
        "RUNWAYML_API_SECRET"
    )
    client = RunwayML(api_key=api_key)
    log.info(
        "creating sound_effect task model=%s duration=%.1fs prompt=%r",
        model, duration_s,
        sfx_prompt[:120] + ("…" if len(sfx_prompt) > 120 else ""),
    )
    task = client.sound_effect.create(
        model=model,                            # type: ignore[arg-type]
        prompt_text=sfx_prompt,
        duration=duration_s,
    )
    task_id = getattr(task, "id", None) or "?"
    log.info("sound_effect task %s polling…", task_id)

    deadline = time.time() + poll_timeout_s
    last_status = ""
    result = None
    while time.time() < deadline:
        t = client.tasks.retrieve(task_id)
        status = getattr(t, "status", None)
        if status != last_status:
            log.info("sound_effect %s status=%s", task_id, status)
            last_status = status or ""
        if status == "SUCCEEDED":
            result = t
            break
        if status in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"sound_effect {status}: {getattr(t, 'failure', None) or status}"
            )
        time.sleep(2.0)
    else:
        raise TimeoutError(
            f"sound_effect task {task_id} did not finish within {poll_timeout_s}s"
        )

    outputs = getattr(result, "output", None) or []
    if not outputs:
        raise RuntimeError("sound_effect task succeeded but returned no output URL")
    output_url = outputs[0] if isinstance(outputs[0], str) else getattr(
        outputs[0], "url", None
    )
    if not output_url:
        raise RuntimeError(f"could not extract output URL: {outputs!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading sound_effect audio → %s", output_path)
    with requests.get(output_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = 0
        with output_path.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    return {"task_id": task_id, "output_url": output_url, "bytes": total}


# ---------------------------------------------------------------------------
# Step 4: ffmpeg mix
# ---------------------------------------------------------------------------


def mix_video_with_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    *,
    music_db: float = -18.0,
) -> None:
    """Mix the video's existing audio (assumed to be the dialogue track)
    with a quieter background music track. Music is attenuated by
    ``music_db`` decibels; ``amix`` then sums them.

    Equivalent ffmpeg: full-volume dialogue + music at ~12% (when -18 dB).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(music_path),
        "-filter_complex",
        # [1:a] = music; lower it; [m] is the quiet music; amix sums veo + music.
        f"[1:a]volume={music_db}dB[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    log.info("ffmpeg mixing → %s (music %.1f dB under dialogue)", output_path, music_db)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("ffmpeg failed (rc=%d):\n%s", proc.returncode, proc.stderr[-2000:])
        raise RuntimeError(f"ffmpeg mix failed (rc={proc.returncode})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composite", required=True, help="Path to the composite still PNG.")
    parser.add_argument("--topic", required=True, help="One-sentence topic for the clip.")
    parser.add_argument(
        "--speech", required=True,
        help="Dialogue spec for Veo (e.g. 'the chrome robot says X, the scholar replies Y')",
    )
    parser.add_argument("--duration", type=int, default=8, choices=[4, 6, 8],
                        help="Video duration (Veo3.1 fast accepts 4/6/8). Music duration matches.")
    parser.add_argument("--ratio", default="1280:720")
    parser.add_argument("--video-model", default="veo3.1_fast")
    parser.add_argument("--music-db", type=float, default=-12.0,
                        help="dB attenuation for music under dialogue (default -12 ≈ 25%%).")
    parser.add_argument("--vibe-model", default="gpt-5.1",
                        help="OpenAI model for music vibe invention.")
    parser.add_argument("--output", help="Final mixed MP4 path. Default: data/movies/<runway-id>_mixed.mp4")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Don't delete the speech-only Veo MP4 and bare music file after mixing.")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    composite = Path(args.composite).expanduser().resolve()
    if not composite.exists():
        print(f"ERROR: composite still not found: {composite}", file=sys.stderr)
        return 2

    # ---- Step 1: vibe ----
    print("=== Step 1: gpt-5.1 invents music vibe ===")
    try:
        vibe = invent_vibe(args.topic, args.speech, model=args.vibe_model)
    except Exception as e:
        print(f"vibe-invent FAILED: {e}", file=sys.stderr)
        return 1
    print(f"  vibe model : {vibe['model_used']}")
    print(f"  vibe       : {vibe['vibe']}")
    print(f"  sfx prompt : {vibe['sfx_prompt']}")
    print()

    # ---- Step 2: Veo speech-only ----
    print("=== Step 2: Veo3.1 fast (audio) — speech only, no music ===")
    veo_prompt = (
        f"Eight-second cinematic shot inside the scene shown in the still. "
        f"{args.speech.strip()} "
        f"IMPORTANT: NO MUSIC AT ALL in the audio track. Just clear spoken "
        f"dialogue plus a quiet ambient room tone (faint workshop/studio "
        f"hum, soft hardware fans). The music will be added separately in "
        f"post; do not include it. Cinematic camera, soft warm/cool rim "
        f"lighting, stylised 3D animation look, no on-screen text."
    )
    try:
        clip = movies.generate_video(
            [composite], veo_prompt,
            model=args.video_model,
            duration=args.duration,
            ratio=args.ratio,
            position_mode="keyframes",
            audio=True,
        )
    except Exception as e:
        print(f"Veo step FAILED: {e}", file=sys.stderr)
        return 1
    print(f"  veo task id : {clip.runway_task_id}")
    print(f"  veo mp4     : {clip.output_path}")
    print()

    # ---- Step 3: sound_effect music ----
    print("=== Step 3: sound_effect generates background music ===")
    music_dir = storage.DATA_DIR / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    music_path = music_dir / f"{clip.runway_task_id}_music.mp3"
    try:
        sfx = generate_music(vibe["sfx_prompt"], float(args.duration), music_path)
    except Exception as e:
        print(f"sound_effect step FAILED: {e}", file=sys.stderr)
        return 1
    print(f"  sfx task id : {sfx['task_id']}")
    print(f"  music file  : {music_path}  ({sfx['bytes']:,} bytes)")
    print()

    # ---- Step 4: ffmpeg mix ----
    print("=== Step 4: ffmpeg mix ===")
    out = (
        Path(args.output).expanduser().resolve() if args.output
        else clip.output_path.with_name(f"{clip.runway_task_id}_mixed.mp4")
    )
    try:
        mix_video_with_music(clip.output_path, music_path, out, music_db=args.music_db)
    except Exception as e:
        print(f"mix step FAILED: {e}", file=sys.stderr)
        return 1
    size = out.stat().st_size
    print(f"  final mp4   : {out}  ({size:,} bytes)")
    print()

    if not args.keep_intermediate:
        # By default, only keep the final mixed MP4. The intermediate
        # speech-only Veo MP4 + bare music are still useful for debugging
        # so the user can opt to keep them with --keep-intermediate.
        try:
            clip.output_path.unlink(missing_ok=True)
            music_path.unlink(missing_ok=True)
            print(f"  cleaned up intermediate files (use --keep-intermediate to keep)")
        except Exception as e:
            log.warning("cleanup failed: %s", e)

    print()
    print("--- success ---")
    print(f"To view: open '{out}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
