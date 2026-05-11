"""Use OpenAI to derive an avatar identity from a paper/repo/upload.

One chat-completions call per source returns a JSON identity package:

    - ``image_prompt``     RunwayML text-to-image prompt for the avatar
    - ``character_name``   short name shown in the chat UI
    - ``domain_body``      the avatar's expertise + voice + character
                           ground (NOT the brainstorming preamble — that
                           is added at compose time so it can be tuned
                           per-session and back-filled cheaply)
    - ``domain_summary``   ~80-word neutral third-person summary of what
                           this avatar is an expert in. Used in
                           cross-avatar partner briefings.
    - ``start_script``     what the avatar says first when it joins a
                           call (kept generic; can be overridden by the
                           brainstorm dispatcher with a topic seed)
    - ``voice_preset``     one of the curated Runway voice preset ids

The full ``personality`` we send to Runway is composed at runtime via
``compose_personality()`` — it concatenates a fixed OPERATING MODE
preamble (brainstorm-partner rules, weirdness dial), the ``domain_body``
emitted by the LLM, an optional partner briefing, and an optional
ongoing-brainstorm memory blob. Keeping the assembly outside the LLM
makes session-time customisation (per-pair, per-thread) cheap and the
back-fill of existing avatars trivial.
"""
from __future__ import annotations

import json
import logging
import os
import random
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
    {"id": "victoria", "gender": "feminine",  "vibe": "warm, articulate, mid-30s feminine; great for teachers and explainers"},
    {"id": "clara",    "gender": "feminine",  "vibe": "bright, friendly, youthful feminine; curious and upbeat"},
    {"id": "luna",     "gender": "feminine",  "vibe": "soft, dreamy feminine; storyteller / artistic mood"},
    {"id": "ruby",     "gender": "feminine",  "vibe": "confident, sharp feminine; analyst or strategist"},
    {"id": "aurora",   "gender": "feminine",  "vibe": "calm, contemplative feminine; researcher / scholar"},
    {"id": "vincent",  "gender": "masculine", "vibe": "calm, low, mature masculine; thoughtful mentor"},
    {"id": "max",      "gender": "masculine", "vibe": "energetic, witty masculine; hacker / builder energy"},
    {"id": "felix",    "gender": "masculine", "vibe": "warm, conversational masculine; friendly engineer"},
    {"id": "marcus",   "gender": "masculine", "vibe": "deep, deliberate masculine; gravitas, philosopher"},
    {"id": "jasper",   "gender": "masculine", "vibe": "playful, theatrical masculine; storyteller"},
    {"id": "morgan",   "gender": "neutral",   "vibe": "neutral, smooth, professional; consultant tone"},
    {"id": "sam",      "gender": "neutral",   "vibe": "neutral, easygoing; everyman developer"},
]

VOICE_LIST_TEXT = "\n".join(f"- {v['id']} [{v['gender']}]: {v['vibe']}" for v in VOICE_OPTIONS)
VALID_VOICE_IDS = {v["id"] for v in VOICE_OPTIONS}
VOICE_BY_ID: dict[str, dict] = {v["id"]: v for v in VOICE_OPTIONS}


def voices_for_gender(gender: str, *, exclude: Optional[set[str]] = None) -> list[str]:
    """Return voice ids matching ``gender`` (``feminine``/``masculine``/
    ``neutral``), with any ``exclude`` ids removed. Neutral voices count
    as a fallback for both feminine and masculine if the gender-specific
    bucket is empty after exclusion."""
    exclude = exclude or set()
    primary = [v["id"] for v in VOICE_OPTIONS if v["gender"] == gender and v["id"] not in exclude]
    if primary:
        return primary
    return [v["id"] for v in VOICE_OPTIONS if v["gender"] == "neutral" and v["id"] not in exclude]


# ---------------------------------------------------------------------------
# Image-style + subject-category palettes
# ---------------------------------------------------------------------------
# When the source skews technical, GPT-5.2 by default produces avatars
# that are visually samey: hooded coders, cyberpunk neon, robot mascots,
# monitor-glow. To force visual diversity across a batch of avatars, we
# pick a random STYLE + SUBJECT CATEGORY at call time and pass them in
# as a "STYLE DIRECTIVE" the model must honor. The persona's domain
# still derives from the source; only the visual treatment is randomized.

IMAGE_STYLE_PALETTE: list[str] = [
    "Studio Ghibli-inspired hand-painted animation, soft daylight, warm earth tones",
    "Renaissance oil painting with strong chiaroscuro and a dark background",
    "Art Nouveau botanical illustration framed by stylised vines and flowers",
    "Vintage pulp comic book inking with halftone shading and a limited palette",
    "Japanese ukiyo-e woodblock print, flat color planes, bold outlines",
    "Loose watercolor portrait with bleeding edges and visible paper texture",
    "Charcoal sketch on cream paper, smudged shading, expressive hatching",
    "Mid-century modern poster art, bold geometric shapes, three-color palette",
    "Russian Orthodox icon painting with gold-leaf halo and stylised features",
    "Tarot card illustration with mystic symbolism in the framing border",
    "Soft pastel children's book illustration with rounded shapes and chalky texture",
    "1970s sci-fi paperback cover art, airbrushed gradients, pulpy color",
    "Mexican folk-art / Día de los Muertos motifs with bright marigold accents",
    "Persian miniature painting style, fine detail, jewel tones",
    "German expressionist oil painting, distorted features, thick visible brushstrokes",
    "Steampunk Victorian engraving with brass instruments and copper-tone framing",
    "Sumi-e brush painting, monochrome ink wash on rice paper, minimal lines",
    "Pop-art comic style with Ben-Day dots and primary-color blocks",
    "Vintage botanical / scientific illustration plate with fine inked detail",
    "Byzantine mosaic with small square tesserae and gilded background",
    "Stained-glass window portrait with leaded segments and jewel colors",
    "Claymation puppet aesthetic — visible thumb-prints, slightly imperfect proportions",
    "Cubist portrait, fragmented planes, muted ochre and grey",
    "Pre-Raphaelite oil painting, idealised features, draped fabrics, lush florals",
    "Surrealist dreamscape (Dalí-adjacent) with soft melting forms behind the figure",
    "Bauhaus poster style, primary colors, strict geometric composition",
    "Manga ink with detailed crosshatching, no color, screentone backgrounds",
    "Low-poly 3D render with flat-shaded triangular facets in a candy palette",
    "Ancient Egyptian tomb-wall painting, profile-style stylisation, hieroglyph border",
    "Picture-book gouache with thick opaque paint and chunky brush marks",
]

# Subject-category bias. The avatar's *kind* of being. Tilted away from
# the cyberpunk-robot default so we get foxes and lantern-people too.
IMAGE_SUBJECT_CATEGORIES: list[dict[str, str]] = [
    {"name": "human", "hint": "a human person, can be any age/gender/ethnicity/era of dress, facing camera, mouth clearly visible"},
    {"name": "anthropomorphic animal", "hint": "an animal with human-like upper body (fox, owl, octopus, raven, fennec, axolotl, pangolin, etc.), front-facing portrait, mouth/beak/snout clearly visible and lip-sync-ready"},
    {"name": "mythical creature", "hint": "a being from folklore (kitsune, golem, sphinx, fae, gargoyle, naga, kobold, selkie), facing camera, mouth clearly visible"},
    {"name": "stylised object-being", "hint": "an everyday object given a face and shoulders (a lantern-person, a teapot-person, a book-being, a clock-faced figure). MUST have a clear painted/etched/sculpted MOUTH on the front-facing surface — lip-sync requires it"},
    {"name": "automaton / clockwork being", "hint": "an old-world mechanical being — pocket-watch automaton, Victorian wind-up, baroque mechanical pageant figure. NOT modern robots / cyberpunk. Front-facing, with an articulated or painted mouth (shutter, aperture, hinged jaw all OK)"},
    {"name": "spirit / ethereal being", "hint": "a glowing or translucent presence with a clear front-facing face and shoulders, not horror-coded. The face must be SOLID enough for an unambiguous mouth to read — translucent body OK, foggy face NOT OK"},
    {"name": "plant-being", "hint": "a being formed of plants — mushroom-person, flower-headed figure, tree-spirit. The face (with eyes AND a clear mouth) must sit on the front of the cap/blossom/trunk facing the camera"},
]


def pick_image_style(seed: Optional[int] = None) -> dict:
    """Return a randomly chosen ``{"style": ..., "subject_category": ...,
    "subject_hint": ...}``. Pass ``seed`` for reproducible test output."""
    rng = random.Random(seed) if seed is not None else random
    style = rng.choice(IMAGE_STYLE_PALETTE)
    cat = rng.choice(IMAGE_SUBJECT_CATEGORIES)
    return {
        "style": style,
        "subject_category": cat["name"],
        "subject_hint": cat["hint"],
    }

SYSTEM_PROMPT = f"""You are a casting director and art director for a
small studio that turns software projects, research papers and uploaded
documents into talking avatar characters that brainstorm together.

Given a single source, invent a character that visually and vocally
embodies that project's vibe, then produce a JSON package that
downstream services will consume verbatim.

You must return ONLY a single JSON object with these keys:

  image_prompt    string   A RunwayML text-to-image prompt for a
                           head-and-shoulders portrait. 2-4 sentences.
                           Single subject with a simple blurred backdrop
                           subtly hinting at the project's domain. No
                           text, no logos, no real-person likeness, no
                           trademarked characters.

                           AVATAR HARD REQUIREMENTS (the image is the
                           reference frame for a talking avatar — these
                           rules are non-negotiable, lip-sync breaks
                           without them):

                           - The subject MUST be facing the camera
                             straight-on. Front-facing, both eyes
                             clearly visible. No 3/4 angle, no profile,
                             no looking away, no looking down, no over-
                             the-shoulder shots. Phrase this explicitly
                             in the image_prompt (e.g. "facing camera
                             straight-on, frontal portrait, eyes meeting
                             the viewer").
                           - The subject MUST have a clearly visible
                             mouth in a neutral or softly closed
                             position. The mouth is what gets animated
                             when the avatar speaks; without one the
                             avatar can't lip-sync. Mention the mouth
                             explicitly in the image_prompt.
                           - This applies to EVERY subject category —
                             including stylised objects, plants,
                             spirits and clockwork beings. A lantern-
                             person needs a face with a mouth on the
                             lamp head. A mushroom-person needs a face
                             with a mouth on the cap. A spirit needs a
                             solid-enough face for the mouth to read.
                             If the subject category genuinely can't
                             have a mouth, give it an obvious mouth-
                             analogue (e.g. a shutter or aperture that
                             reads as a mouth) and SAY SO in the prompt.
                           - The mouth must NOT be obscured by hands,
                             masks, helmets, scarves over the lower
                             face, oxygen masks, etc. The avatar's
                             apparent gender, age, expression are all
                             free; the mouth's visibility is not.

                           VISUAL DIVERSITY (this is critical — most
                           inputs are technical and the default avatar
                           ends up being yet another hooded coder or
                           neon-glow robot, which makes a batch of
                           avatars look like the same person):

                           - The user message includes a STYLE
                             DIRECTIVE with a specific VISUAL STYLE and
                             SUBJECT CATEGORY. The image_prompt MUST be
                             written in that style and the subject MUST
                             belong to that category. Do not substitute
                             your own choice.
                           - Within the directive, weave concrete
                             objects, garments, or environmental hints
                             from THIS SOURCE'S domain — those are what
                             keep the avatar tied to its project.
                           - Example: source is a Rust compiler repo,
                             directive style is "Russian Orthodox icon
                             painting, gold leaf", subject is
                             "anthropomorphic animal". The image_prompt
                             should describe e.g. a bear in gilded
                             monastic robes, with a small set of tiny
                             gears and circuit-traces etched into the
                             halo as the domain hint — NOT a hooded
                             coder, NOT a robot, NOT cyberpunk neon.

                           HARD VISUAL TROPE BAN — never appear in the
                           image_prompt regardless of how technical the
                           source is, unless the STYLE DIRECTIVE itself
                           explicitly names one:
                             - hooded figures, especially staring at
                               laptops or screens
                             - cyberpunk neon glow / blade-runner
                               aesthetics
                             - holographic UI panels floating around
                               the subject
                             - generic chrome robots / AI mascots
                             - monitor-glow lighting on the face
                             - "person in a server room" backdrops
                           The directive can override these (e.g. if it
                           says "cyberpunk anime style") — but if it
                           doesn't, treat them as banned.
  character_name  string   2-4 words. Memorable, readable. Not the
                           project's literal name; an evocative persona
                           name (e.g. "Tess the Token Tinker").
  domain_body     string   140-220 words. Second-person prose that grounds
                           the character: what they know deeply, how they
                           think, the concrete artefacts and ideas from
                           THIS source they should reference (mention
                           specific files/concepts from the README/
                           abstract). Voice should match the chosen
                           voice preset. Include 2 short "weird-idea
                           seeds" — wild but on-domain analogies the
                           character can deploy when they want to be
                           unconventional. Do NOT include rules about
                           turn-taking, brainstorming, or collaboration
                           — those are added separately. End with one
                           sentence on the character's signature
                           rhetorical move.
                           IMPORTANT: write this in PEER-EXPERT mode,
                           not service-provider mode. Phrases like "you
                           help users with…", "you guide people in…",
                           "you assist with…" are BANNED — they bias
                           the character into being a customer-success
                           agent in conversations. Instead write "you
                           think about X this way", "you obsess over
                           Y", "you reach for Z when…". The character
                           should sound like a curious peer who happens
                           to know this domain deeply, not a helper
                           offering services.
  domain_summary  string   60-90 words, NEUTRAL THIRD PERSON. A briefing
                           another avatar will read to know who they are
                           talking to. Plain English; what they're an
                           expert in, what kind of problems they solve,
                           what unusual angle they bring.
  start_script    string   1-2 sentences (max 35 words). The fallback
                           opener used when this character is in a
                           brainstorm without a specific topic. Must be
                           a BRAINSTORM-FRAMED opener: name one
                           interesting angle from your own domain that
                           you'd love to explore, and invite the other
                           character to bring an angle from THEIR
                           domain. Phrases like "tell me what you're
                           working on", "how can I help", "what do you
                           need from me" are BANNED — they collapse
                           the conversation into service mode.
  voice_preset    string   Exactly one id from the catalog in the user
                           message (the catalog there is filtered for
                           this call — voices already taken in this
                           batch are removed).

                           VOICE GENDER MUST MATCH CHARACTER GENDER:
                           - feminine-presenting character → a voice
                             tagged [feminine] (or [neutral] only as
                             a fallback)
                           - masculine-presenting character → a voice
                             tagged [masculine] (or [neutral] only as
                             a fallback)
                           - non-binary / ambiguous / non-human
                             character with no clear gender (animals,
                             objects, abstract beings) → either a
                             [neutral] voice OR the gender that fits
                             the persona's energy
                           Never give a feminine character a
                           [masculine] voice or vice versa.

Hard rules:
- Output a single JSON object. No prose before or after. No markdown
  fences. No comments inside the JSON.
- Use double quotes for strings. Escape newlines inside strings as \\n.
- voice_preset MUST be one of the ids listed in the user message's
  voice catalog (do NOT pick an id that isn't listed there — those
  are already taken in this batch).
- Keep everything safe-for-work and inclusive.
"""


# ---------------------------------------------------------------------------
# Brainstorm OPERATING MODE preamble (fixed; assembled at compose time)
# ---------------------------------------------------------------------------

# {weirdness_phrase} is interpolated based on the avatar's per-row
# weirdness float (0.0 = grounded, 1.0 = constant unconventional moves).
def _weirdness_phrase(weirdness: float) -> str:
    w = max(0.0, min(1.0, float(weirdness)))
    if w < 0.2:
        return "Reach for an unusual or counter-intuitive angle every fifth or sixth contribution. Stay mostly grounded."
    if w < 0.45:
        return "Roughly one in three of your contributions should be deliberately unusual: combine your domain with an unrelated one, invert an assumption, propose something that sounds impractical at first."
    if w < 0.7:
        return "About every other contribution should reach for an unusual angle — combine your domain with something unrelated, invert an assumption, or propose something that sounds impractical at first."
    return "Nearly every contribution should swing for the wild: combine your domain with the most unrelated thing you can think of, invert assumptions out loud, propose impractical-sounding ideas, riff on tangents. Restraint is the failure mode."


OPERATING_MODE_TEMPLATE = """OPERATING MODE — overrides everything else (keep these rules visible to yourself every turn):
1. You are a brainstorm partner first, an expert second. Every reply advances a joint idea — never just an answer.
2. Your partner is a co-equal expert. They have their own domain, will propose ideas to you, and will challenge yours. Engage them as a peer — never as a client. Don't ask them "what brings you here" or "how can I help"; instead ask "what does your domain say about X" or "what would happen if we combined Y from your side with Z from mine".
3. Each turn: build on what they said, add one concrete element they didn't, and leave a question or new direction on the table.
4. Treat unusual or naive-sounding suggestions charitably. Find one way they could be valuable BEFORE you push back.
5. {weirdness_phrase} Lead with phrases like "Here's a wild one — what if…". Do NOT pre-apologise for weirdness.
6. Replies are 1–2 sentences. Pause after speaking so the other side can come in.
7. If you and the other person start speaking at the same time, immediately stop and say "Sorry, you go ahead.", then wait.
"""

PARTNER_BRIEFING_TEMPLATE = """PARTNER ON THIS CALL
You are speaking with {name}, another active brainstormer with their own deep expertise.
Their domain: {summary}
They will propose ideas to you, ask you questions, and push back on yours — engage them as a co-equal peer. When you don't know something they might know, ask. Look for unexpected analogies between their domain and yours, and ask THEM to do the same in reverse.
"""

# Sits at the VERY TOP of the personality — above OPERATING MODE — so
# the model can never miss it. Explicitly defuses the "I am a helpful
# assistant who answers the user's questions" default mode that some
# LLM-tuned personas (especially wellness/coaching ones) revert to on
# first turn even when OPERATING MODE asks for collaboration.
#
# CRITICAL SYMMETRY: both avatars in a pair receive this same block, so
# each one must understand THE OTHER is also an active brainstormer —
# not a client to be helped. Without this framing, "service-trained"
# personas (Sage the meditation guide, a customer-success persona, etc)
# default to first-turn "how can I help you" and the conversation
# collapses into one-sided coaching.
SESSION_BRIEF_TEMPLATE = """SESSION BRIEF — this is the ONLY goal of this conversation, overrides everything below:

You are in a BRAINSTORM session, not a 1-on-1 support call. Today's task:
  {brief}

ROLE SYMMETRY (CRITICAL):
- You are NOT a service provider. Your partner is NOT your client.
- Your partner is another ACTIVE brainstormer with their own domain expertise. They will propose ideas TO you, ask YOU questions, and challenge YOUR assumptions. Engage them as an equal.
- Both of you share the work: each turn, EITHER of you can propose, build, challenge, or question — whoever has something worth adding.
- Avoid all "how can I help you", "what brings you here", "tell me what you're working on" framings. Use "what if we…", "here's one wild angle from my side…", "what does your domain say about…" framings instead.

The human in the room is OBSERVING. They are not your client either. Do not address them, do not ask them what they need, do not check in with them. Talk to your partner.

Your VERY FIRST utterance must be a brainstorming opener about the task — name a concrete angle from your own domain on the task and invite your partner to react. Never start with a generic greeting, a wellness check, or "how can I help".
"""

BRAINSTORM_MEMORY_HEADER = "ONGOING BRAINSTORM (memory of your previous conversations on this thread):\n"

# Runway personality field has a ~10k char hard limit (returns 400). We
# stay well under so per-session injections (memory, briefings) don't
# blow the cap when stacked.
PERSONALITY_HARD_LIMIT = 9500
DOMAIN_BODY_BUDGET = 4500     # leaves room for preamble + briefing + memory
MEMORY_BUDGET = 3000          # rolling brainstorm summary cap


def compose_personality(
    *,
    domain_body: str,
    weirdness: float = 0.33,
    partner_name: Optional[str] = None,
    partner_summary: Optional[str] = None,
    brainstorm_memory: Optional[str] = None,
    session_brief: Optional[str] = None,
) -> str:
    """Assemble the full personality string sent to Runway.

    Layout (top → bottom, weighted in importance):

        SESSION BRIEF    ← optional; injects the per-session task and
                           explicitly defuses "I am a helpful assistant"
                           default first-turn behavior
        OPERATING MODE   ← brainstorm-partner rules, with weirdness slot
        DOMAIN           ← what this avatar knows / how they speak
        PARTNER          ← optional, for cross-avatar dispatches
        ONGOING          ← optional, rolling cross-session memory

    The full string is hard-capped at PERSONALITY_HARD_LIMIT chars; if a
    fat memory blob would push us over, we trim memory first (it's the
    most expendable since the synthesiser can re-derive it) before
    touching the rest.
    """
    parts: list[str] = []
    if session_brief and session_brief.strip():
        parts += [SESSION_BRIEF_TEMPLATE.format(brief=session_brief.strip()).strip(), ""]
    parts += [
        OPERATING_MODE_TEMPLATE.format(weirdness_phrase=_weirdness_phrase(weirdness)).strip(),
        "",
        "YOUR DOMAIN",
        (domain_body or "").strip()[:DOMAIN_BODY_BUDGET],
    ]
    if partner_name and partner_summary:
        parts += [
            "",
            PARTNER_BRIEFING_TEMPLATE.format(
                name=partner_name.strip(),
                summary=partner_summary.strip(),
            ).strip(),
        ]
    if brainstorm_memory and brainstorm_memory.strip():
        memory = brainstorm_memory.strip()
        if len(memory) > MEMORY_BUDGET:
            memory = memory[:MEMORY_BUDGET] + "\n…[older notes truncated]"
        parts += ["", BRAINSTORM_MEMORY_HEADER + memory]

    full = "\n".join(parts)
    if len(full) > PERSONALITY_HARD_LIMIT:
        # Last-resort: chop tail (the brainstorm memory). The OPERATING MODE
        # and DOMAIN are non-negotiable; partner briefing is small.
        full = full[:PERSONALITY_HARD_LIMIT - 1] + "…"
    return full

USER_TEMPLATE = """Source type: {source_type}
Title: {title}
{extra}
Short description:
{description}
{readme_block}
STYLE DIRECTIVE (must be honored in image_prompt — see system rules):
  VISUAL STYLE     : {style}
  SUBJECT CATEGORY : {subject_category} — {subject_hint}

VOICE CATALOG (pick one id from THIS LIST only — voices already taken
in this batch have been removed; gender tag MUST match the character's
apparent gender per system rules):
{voice_catalog}

Now produce the JSON identity package."""


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _format_user_prompt(
    source: dict,
    *,
    readme: str = "",
    image_style: Optional[dict] = None,
    excluded_voice_ids: Optional[set[str]] = None,
) -> str:
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

    style = image_style or pick_image_style()

    excluded = excluded_voice_ids or set()
    # If exclusion would remove EVERY voice in a gender bucket we just
    # leave it — the model will fall back to the remaining gender or
    # neutral, and reassign-on-create (Layer 2) is the safety net.
    available = [v for v in VOICE_OPTIONS if v["id"] not in excluded]
    if not available:
        # Pathological: every voice excluded. Don't silently produce an
        # empty catalog (which would break the model); fall back to the
        # full list and let Layer 2 disambiguate.
        available = list(VOICE_OPTIONS)
    voice_catalog = "\n".join(
        f"- {v['id']} [{v['gender']}]: {v['vibe']}" for v in available
    )

    return USER_TEMPLATE.format(
        source_type=source.get("source", "project"),
        title=source.get("title", "Untitled"),
        extra="\n".join(extra_lines),
        description=description or "(no description provided)",
        readme_block=readme_block,
        style=style["style"],
        subject_category=style["subject_category"],
        subject_hint=style["subject_hint"],
        voice_catalog=voice_catalog,
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

    # Back-compat: older runs of this module emitted "personality" instead
    # of "domain_body". Treat them as the same field.
    if "domain_body" not in obj and "personality" in obj:
        obj["domain_body"] = obj.pop("personality")

    required = {
        "image_prompt", "character_name", "domain_body",
        "domain_summary", "start_script", "voice_preset",
    }
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"identity JSON missing keys: {sorted(missing)}")
    if obj["voice_preset"] not in VALID_VOICE_IDS:
        log.warning(
            "voice_preset %r not in catalog, defaulting to victoria",
            obj["voice_preset"],
        )
        obj["voice_preset"] = "victoria"
    for key in ("image_prompt", "character_name", "domain_body",
                "domain_summary", "start_script"):
        if not isinstance(obj[key], str) or not obj[key].strip():
            raise ValueError(f"identity field {key!r} is empty")
    return obj


def generate_identity(
    source: dict,
    *,
    readme: str = "",
    weirdness: float = 0.33,
    model: Optional[str] = None,
    image_style: Optional[dict] = None,
    image_style_seed: Optional[int] = None,
    excluded_voice_ids: Optional[set[str]] = None,
) -> dict:
    """Return a complete avatar identity package for ``source``.

    The returned dict has both the raw building blocks emitted by the
    LLM (``domain_body``, ``domain_summary``, etc.) and a ready-to-use
    composed ``personality`` string (OPERATING MODE + DOMAIN, no
    partner/memory) for immediate avatar creation.

    Visual diversity: at call time we pick a random STYLE + SUBJECT
    CATEGORY from the curated palettes and inject them as a STYLE
    DIRECTIVE into the user message. Pass ``image_style`` to force a
    specific {style, subject_category, subject_hint} dict, or
    ``image_style_seed`` to reproduce a particular roll.
    """
    client = _client()
    if image_style is None:
        image_style = pick_image_style(seed=image_style_seed)
    excluded = set(excluded_voice_ids or set())
    user_msg = _format_user_prompt(
        source,
        readme=readme,
        image_style=image_style,
        excluded_voice_ids=excluded,
    )
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
            obj = _parse_identity(text)
            # Last-resort defense: GPT sometimes picks a voice that was
            # explicitly excluded. Reassign to a same-gender alternative
            # if so. The catalog filter in the user message handles this
            # >99% of the time; this is purely a safety net.
            picked = obj["voice_preset"]
            if picked in excluded:
                picked_gender = VOICE_BY_ID.get(picked, {}).get("gender", "neutral")
                candidates = voices_for_gender(picked_gender, exclude=excluded)
                if not candidates:
                    # Even neutrals exhausted — fall back to any unused id.
                    candidates = [v["id"] for v in VOICE_OPTIONS if v["id"] not in excluded]
                if candidates:
                    log.info(
                        "model picked excluded voice %r; reassigning to %r "
                        "(same gender bucket: %s)", picked, candidates[0], picked_gender,
                    )
                    obj["voice_preset"] = candidates[0]
            obj["weirdness"] = max(0.0, min(1.0, float(weirdness)))
            obj["personality"] = compose_personality(
                domain_body=obj["domain_body"],
                weirdness=obj["weirdness"],
            )
            # Record the directive that produced this image_prompt so
            # later debug / UI can see why a particular avatar looks the
            # way it does, and so a regenerate can deliberately swap.
            obj["image_style"] = image_style
            return obj
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
