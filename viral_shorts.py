"""Turn one YouTube URL into a captioned, hook-driven 9:16 Short, hands-off.

Pipeline: smart URL ingestion (video, Shorts, livestream, or channel link ->
newest upload) -> subtitle-only transcript (no video bytes) -> the AI picks
high-retention 35-50 second windows that open on the hook. With GEMINI_API_KEY
set the Google Gemini API answers directly (generate_text_gemini), bypassing
the OpenCode free quota entirely; without a key the local OpenCode models run
(MiMo with automatic fallback to Nemotron 3.5 Super VL and DeepSeek V4.1
Flash). By default up to 5 DISTINCT moments are found; only the genuinely
strong, well-spread ones ship (1..N - a video with one great moment returns one
clip, never filler) -> yt-dlp cuts only those sections -> FFmpeg
crops to 1080x1920 and burns high-contrast captions plus the AI-suggested
on-screen text hook, and grabs each clip's peak-payoff frame as a cover
reference -> every clip is saved to instagram_manual_ready/ with its JSON
metadata for manual Instagram upload and published to YouTube as a Short with
AI-written, high-SEO metadata (title <60 chars ending in #Shorts, six fixed
hashtags), sequentially with a safe upload gap.

Low-spec safety (i5 5th gen / GT 730 class): every yt-dlp and FFmpeg command
runs with a hard timeout and no stdin, x264 uses the veryfast preset with
cheap bicubic scaling, and --threads / --x264-preset let you trade quality for
speed. Nothing can hang the terminal or exhaust memory.

The Instagram path is manual by design. The YouTube upload runs only against
the channel authorized in client_secrets.json; pass --skip-upload to stop before
it. Keys and tokens stay in local files and are never printed.

Unattended use: pass only the URL and the script runs end to end with no
prompts and publishes by default (one-time browser consent writes
youtube_token.json; runs without a terminal never open a browser and fail with
instructions instead):

    python viral_shorts.py "https://www.youtube.com/watch?v=..."
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "opencode/mimo-v2.6-flash-free"
# automatic quota-fallback chain, tried in order when the free limit is hit
FALLBACK_MODELS = (
    "opencode/nemotron-3.5-super-vl-free",
    "opencode/deepseek-v4.1-flash-free",
)
DEFAULT_OUTPUT_DIR = "instagram_manual_ready"
CLIENT_SECRETS = HERE / "client_secrets.json"
YOUTUBE_TOKEN = HERE / "youtube_token.json"
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
HASHTAGS = ["#Shorts", "#Gaming", "#GTA6", "#Viral", "#Trending", "#GamerMoments"]
SEGMENT_MIN = 35
SEGMENT_MAX = 50
TITLE_MAX = 60           # high-CTR Shorts titles stay short and must carry #Shorts
HOOK_MAX = 40            # burned-in on-screen hook text (plain ASCII only)
CLIPS_MAX = 5            # most distinct moments one run will extract
CLIP_SCORE_FLOOR = 65    # a candidate below this is never shipped
UPLOAD_GAP_DEFAULT = 90  # seconds between sequential uploads (kind to the API)
CLIPS_DEFAULT = 5        # a bare `viral_shorts.py URL` asks for up to 5 genuine moments
CLIPS_DEFAULT = 5        # a bare `viral_shorts.py URL` ships 3..5 clips by quality
# ---------- low-spec safety (i5 5th gen / GT 730 class hardware) ----------
# Every external command is bounded in time so a stalled network or a slow
# encode can never freeze the terminal or the machine.
X264_PRESET = "veryfast"  # fast encode + small files; the sweet spot for old CPUs
X264_CRF = 20             # visually clean at 1080x1920 without wasting CPU
SCALER_FLAGS = "bicubic"  # far cheaper than lanczos, visually close at this size
YTDLP_TIMEOUT = 180       # metadata / subtitle / channel lookups (network-bound)
SECTION_TIMEOUT = 600     # one 35-50s section download
RENDER_TIMEOUT = 1800     # x264 encode of one vertical clip
COVER_TIMEOUT = 60        # single frame grab
WINDOW_CHARS = 60000       # transcript characters per model call
WINDOW_OVERLAP_S = 60      # seconds of context repeated across windows
JSON_TRIES = 3
API_TIMEOUT = (15, 900)    # (connect, read) seconds
YOUTUBE_TAG_BUDGET = 480   # YouTube allows 500 characters across all tags
SERVICE_PATHS = (
    lambda: Path(os.environ["XDG_STATE_HOME"]) / "opencode" / "service.json" if os.environ.get("XDG_STATE_HOME") else None,
    lambda: Path.home() / ".local" / "state" / "opencode" / "service.json",
    lambda: Path(os.environ["LOCALAPPDATA"]) / "opencode" / "service.json" if os.environ.get("LOCALAPPDATA") else None,
)
FFMPEG_HINT = (
    "FFmpeg was not found. Install it and keep it on PATH, for example:\n"
    "  choco install ffmpeg   (Chocolatey)\n"
    "  scoop install ffmpeg   (Scoop)\n"
    "or download the full build from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder to PATH."
)


class Fail(Exception):
    """A user-facing error with a clear message."""


# ---------- small helpers ----------
def fmt_ts(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_timestamp(value):
    """Accept 57, '57', '15:20', '00:15:20', '00:15:20.500' and return seconds."""
    if isinstance(value, bool):
        raise ValueError(f"bad timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {value!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"bad timestamp: {value!r}") from exc
    if len(nums) == 3:
        hours, minutes, secs = nums
    else:
        hours, minutes, secs = 0.0, nums[0], nums[1]
    if minutes >= 60 or secs >= 60:
        raise ValueError(f"bad timestamp: {value!r}")
    return hours * 3600 + minutes * 60 + secs


def normalize_segment(start, end, duration, min_len=SEGMENT_MIN, max_len=SEGMENT_MAX):
    """Force the segment into [min_len, max_len] seconds inside the video."""
    start_i = max(0, int(round(start)))
    try:
        length = int(round(float(end) - float(start)))
    except (TypeError, ValueError):
        length = min_len
    length = max(min_len, min(max_len, length))
    if duration and duration > 0:
        dur_i = int(duration)
        if dur_i <= min_len:
            return 0, max(1, dur_i)
        if start_i + length > dur_i:
            start_i = dur_i - length
        start_i = max(0, start_i)
        length = min(length, dur_i - start_i)
    return start_i, start_i + max(1, length)


def trim_words(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") or text[:limit].rstrip(" ,;:-")


def sanitize_filename(text, limit=60):
    text = re.sub(r"[^A-Za-z0-9 _-]+", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return trim_words(text, limit)


def extract_json(text):
    """Pull the first JSON object out of a model reply (fences or prose allowed)."""
    text = str(text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        idx = candidate.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(candidate[idx:])
            except ValueError:
                idx = candidate.find("{", idx + 1)
                continue
            if isinstance(obj, dict):
                return obj
            idx = candidate.find("{", idx + 1)
    raise ValueError("reply contained no JSON object")


def cue_line(cue):
    return f"[{fmt_ts(cue['start'])}] " + " ".join(str(cue["text"]).split())


def transcript_block(cues):
    return "\n".join(cue_line(c) for c in cues)


def chunk_cues(cues, max_chars=WINDOW_CHARS):
    """Split the transcript into model-sized windows with a small time overlap."""
    windows, current, size = [], [], 0
    for cue in cues:
        line = cue_line(cue) + "\n"
        if current and size + len(line) > max_chars:
            windows.append(current)
            cutoff = current[-1]["start"] - WINDOW_OVERLAP_S
            current = [c for c in current if c["end"] > cutoff]
            size = sum(len(cue_line(c)) + 1 for c in current)
        current.append(cue)
        size += len(line)
    if current:
        windows.append(current)
    return windows


def section_range(start, end):
    return f"*{fmt_ts(start)}-{fmt_ts(end)}"


def build_prompt(transcript, window_index, window_total, video_title, duration, count=1):
    """Build the analyst prompt; count=1 asks for one clip, count>1 for a ranked list."""
    length_hint = f"The video is {fmt_ts(duration)} long." if duration else "Duration is unknown (live stream)."
    window_hint = ""
    if window_total > 1:
        window_hint = (
            f"\nThis is part {window_index} of {window_total} of one transcript. "
            "Only pick a segment whose timestamps fall inside the lines you were given."
        )
    clip_fields = (
        '  "start": "HH:MM:SS",\n'
        '  "end": "HH:MM:SS",\n'
        '  "peak": "HH:MM:SS of the single most intense second inside the window",\n'
        '  "title": "high-CTR curiosity-driven clickbait title under 60 characters, ending with #Shorts",\n'
        '  "description": "high-SEO keyword-rich description",\n'
        f'  "hashtags": {json.dumps(HASHTAGS)},\n'
        '  "hook": "short text hook to burn on screen, plain ASCII, max 40 characters, no emoji",\n'
        '  "tags": ["8 to 15 SEO keywords without the # sign"],\n'
        '  "score": 0,\n'
        '  "reason": "one sentence"\n'
    )
    if count > 1:
        schema = (
            "Reply with ONLY one JSON object. No markdown fence, no commentary, no extra keys. "
            "The clips array holds one object per clip - repeat the whole object for each clip, "
            "ranked best first. Schema (one clip object shown):\n"
            "{\n  \"clips\": [\n    {\n" + clip_fields + "    }\n  ]\n}\n"
        )
        count_rules = (
            f"- return ONE clip object per distinct strong moment, up to {count} total, "
            "ranked best first. Scan the ENTIRE transcript before choosing; long gaming "
            "videos usually contain several distinct strong moments, so aim for the "
            "maximum when they genuinely exist.\n"
            "- never pad the list: return fewer clips rather than weak, repetitive, or "
            "overlapping filler.\n"
            "- include a clip only if you would score it 65 or higher.\n"
            "- clips must be DISTINCT: zero overlapping timestamps and at least 5 seconds apart.\n"
            "- spread the clips across the whole video (beginning, middle, and end), never "
            "clustered around one moment.\n"
            "- every clip must open immediately on its own strong visual or verbal hook.\n"
            "- every title must be unique and specific to its clip; never reuse a formula.\n"
        )
    else:
        schema = (
            "Reply with ONLY one JSON object. No markdown fence, no commentary, no extra keys. Schema:\n"
            "{\n" + clip_fields + "}\n"
        )
        count_rules = ""
    if count > 1:
        focus = (
            "You are MiMo Shorts Analyst, a viral YouTube Shorts strategist for a gaming channel "
            "whose clips routinely pass millions of views. From the transcript below, choose the "
            "highest-retention moments - one clip per distinct strong scene. Each clip must peak on "
            "emotion first (a scream, a clutch play, an insane fail, a huge reveal, a real "
            "laugh-out-loud moment), and every cut must open on its hook, never on dead air or a "
            "slow build-up. Shortlist at least six candidate moments spread through the whole "
            "transcript before keeping the strongest ones.\n"
        )
    else:
        focus = (
            "You are MiMo Shorts Analyst, a viral YouTube Shorts strategist for a gaming channel "
            "whose clips routinely pass millions of views. From the transcript below, choose the "
            "SINGLE highest-retention moment: peak emotion first (a scream, a clutch play, an insane "
            "fail, a huge reveal, a real laugh-out-loud moment), and the cut must open on the hook, "
            "never on dead air or a slow build-up.\n"
        )
    return (
        "Analyze this gaming stream transcript like a pro viral short-form editor. "
        "Look for high-energy moments, unexpected shock value, sudden screams, funny fails, "
        "or climax hype. Pick a tight "
        f"{SEGMENT_MIN}-{SEGMENT_MAX} second window that starts immediately with a strong "
        "visual or verbal hook.\n"
        + focus +
        schema +
        "Rules:\n"
        f"- end minus start must be between {SEGMENT_MIN} and {SEGMENT_MAX} seconds.\n"
        "- timestamps are absolute video time in HH:MM:SS and must lie inside the transcript timeline.\n"
        "- the window must open immediately on the strong visual or verbal hook described above.\n"
        "- peak: the exact second of the biggest payoff inside the window (scream, impact, reveal); "
        "it is used to pick the thumbnail frame.\n"
        f"- title: curiosity-driven high-CTR clickbait for algorithm push (shock-value hooks, "
        "'GTA 6 moments' style), {TITLE_MAX} characters or fewer INCLUDING a trailing '#Shorts', "
        "and it must describe only what actually happens in the transcript.\n"
        "- description: high-SEO, keyword-rich, truthful; no invented events, reactions, creator "
        "approval, or viewership numbers.\n"
        f"- hashtags: exactly these six strings: {' '.join(HASHTAGS)}\n"
        f"- hook: the punchy on-screen text burned onto the video frames for the whole clip "
        f"(e.g. 'WAIT FOR IT...'), plain English ASCII only, {HOOK_MAX} characters max, no emoji, "
        "no quotation marks - emoji cannot be rendered into the frames.\n"
        "- tags: 8-15 English SEO keywords, no '#' prefix.\n"
        "- score: 0-100 viral potential of this exact window.\n"
        "- Write title, description, hook, and tags in English even if the transcript is another language.\n"
        + count_rules +
        f"{length_hint}{window_hint}\n"
        "TRANSCRIPT:\n"
        f"{transcript}"
    )


def validate_analysis(obj):
    missing = [k for k in ("start", "end", "title", "description") if not obj.get(k)]
    if missing:
        raise ValueError("missing keys: " + ", ".join(missing))
    parse_timestamp(obj["start"])
    parse_timestamp(obj["end"])
    return obj


def ensure_shorts_title(title):
    """High-CTR title contract: <= TITLE_MAX chars, always carrying #Shorts."""
    title = trim_words(title, TITLE_MAX) or "Insane moment you have to see"
    if "#shorts" not in title.lower():
        title = (trim_words(title, TITLE_MAX - len(" #Shorts")) + " #Shorts").strip()
    return title


def normalize_hook(raw):
    """Screen-hook contract: printable ASCII words only, so it burns cleanly."""
    text = " ".join(str(raw or "").split())
    text = "".join(ch for ch in text if " " <= ch <= "~")   # drop emoji/other scripts
    text = text.replace('"', "").replace("'", "")           # quote marks look wrong on frames
    return trim_words(text, HOOK_MAX)


def enforce_analysis(raw, duration):
    """Clamp the model's pick and force the metadata contract the upload needs."""
    start = parse_timestamp(raw["start"])
    end = parse_timestamp(raw["end"])
    start_i, end_i = normalize_segment(start, end, duration)
    title = ensure_shorts_title(raw.get("title"))
    description = " ".join(str(raw.get("description") or "").split())
    missing_tags = [tag for tag in HASHTAGS if tag not in description]
    if missing_tags:
        description = (description + "\n\n" + " ".join(missing_tags)).strip()
    tags = []
    seen = set()
    for item in raw.get("tags") or []:
        tag = str(item).strip().lstrip("#").strip()
        key = tag.lower()
        if tag and key not in seen:
            seen.add(key)
            tags.append(tag)
    if not tags:
        tags = ["gaming", "youtube shorts", "viral clips", "gameplay", "highlights"]
    kept, budget = [], YOUTUBE_TAG_BUDGET
    for tag in tags:
        if len(tag) + 1 > budget:
            break
        kept.append(tag)
        budget -= len(tag) + 1
        if len(kept) >= 15:
            break
    try:
        score = float(raw.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    try:  # thumbnail frame: the biggest payoff second, always inside the window
        peak = parse_timestamp(raw.get("peak"))
    except (ValueError, TypeError):
        peak = (start_i + end_i) / 2
    peak = int(round(min(max(peak, start_i), end_i)))
    return {
        "start": start_i,
        "end": end_i,
        "peak": peak,
        "title": title,
        "description": description,
        "hashtags": list(HASHTAGS),
        "hook": normalize_hook(raw.get("hook")),
        "tags": kept,
        "score": score,
        "reason": " ".join(str(raw.get("reason") or "").split()),
    }


def provider_error_message(status, body):
    """Turn an OpenCode provider error body into a clear user message."""
    reason = message = None
    retry_ms = None
    try:
        data = json.loads(body)
        reason = data.get("reason")
        message = data.get("message")
        retry_ms = data.get("retryAfterMs")
    except (ValueError, AttributeError):
        pass
    if reason == "rate_limit" or (message and "usage limit" in message.lower()):
        reset = ""
        if isinstance(retry_ms, (int, float)) and retry_ms > 0:
            when = datetime.now().astimezone() + timedelta(milliseconds=retry_ms)
            reset = f" It resets around {when.strftime('%Y-%m-%d %H:%M %Z')}."
        return (
            "OpenCode's free usage limit for free models is reached." + reset +
            " Wait for the reset or subscribe to OpenCode Go: https://opencode.ai/go"
        )
    if message:
        return f"OpenCode API error {status}: {message}"
    return f"OpenCode API error {status}: {str(body or '')[:300]}"


# ---------- OpenCode API (MiMo) ----------
def opencode_service():
    url = os.environ.get("OPENCODE_SERVER_URL")
    password = os.environ.get("OPENCODE_PASSWORD")
    if url:
        return url.rstrip("/"), password or ""
    for make in SERVICE_PATHS:
        path = make()
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise Fail(f"could not read OpenCode service file {path}: {exc}") from exc
            return str(data.get("url", "")).rstrip("/"), str(data.get("password") or "")
    raise Fail(
        "OpenCode service file not found. Open the OpenCode app (or run 'opencode service start') "
        "so it can register, then retry. You can also set OPENCODE_SERVER_URL and OPENCODE_PASSWORD."
    )


def parse_model_ref(ref):
    ref = str(ref or "").strip()
    variant = None
    if "#" in ref:
        ref, variant = ref.split("#", 1)
    provider, _, model_id = ref.partition("/")
    if not provider or not model_id:
        raise Fail(f"--model must look like provider/model, got: {ref!r}")
    payload = {"providerID": provider, "id": model_id}
    if variant:
        payload["variant"] = variant
    return payload


def list_free_models(url, password):
    try:
        response = requests.get(f"{url}/api/model", auth=("opencode", password), timeout=60)
        response.raise_for_status()
        models = response.json().get("data") or []
    except (requests.RequestException, ValueError):
        return []
    free = []
    for model in models:
        cost = (model.get("cost") or [{}])[0]
        if not cost.get("input") and not cost.get("output"):
            free.append(f"{model.get('providerID')}/{model.get('modelID')}")
    return free[:12]


# ---------- Google Gemini bypass (independent of the OpenCode free quota) ----------
# Plain REST with the existing `requests` dependency - no extra package needed.
# The key travels only in the x-goog-api-key header, never in the URL, so it can
# never leak into an exception message or a log line.
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"   # current stable flash model (see ai.google.dev/gemini-api/docs/models)
GEMINI_TIMEOUT = (10, 300)   # (connect, read) seconds per request; a 60k-char prompt takes ~4 min
GEMINI_BUDGET = 900          # hard ceiling for one generate_text() call, across all models
GEMINI_ROUNDS = 2            # retry the whole ladder once if every model was overloaded
GEMINI_RETRY_PAUSE = 20      # seconds to wait before that second pass
_GEMINI_ANSWERED_MODEL = ""  # the id that actually answered, for accurate metadata
# tried in order when a model is overloaded (429/500/502/503), stalls, or is missing
GEMINI_MODEL_LADDER = (
    "gemini-3.8-flash",        # default; occasionally 503 under load
    "gemini-3.6-flash",        # most reliable for long transcripts
    "gemini-3.5-flash-lite",   # lighter and fast
    "gemini-3.1-flash-lite",   # last resort; consistently available
)


GEMINI_KEY_ENV = "GEMINI_API_KEY"


def clean_key(raw):
    """Normalize a key: trim whitespace/newlines and accidental surrounding quotes."""
    return str(raw or "").strip().strip('"').strip("'").strip()


def _windows_persistent_env(name):
    """Read a User/Machine environment variable straight from the Windows registry.

    A process environment is a snapshot taken at launch, so a key set later with
    ``[Environment]::SetEnvironmentVariable(..., "User")`` is invisible to
    ``os.environ``. Reading the registry makes the check miss-proof.
    """
    if os.name != "nt":
        return ""
    try:
        import winreg
    except ImportError:
        return ""
    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, subkey in locations:
        try:
            with winreg.OpenKey(root, subkey) as handle:
                value, _ = winreg.QueryValueEx(handle, name)
        except OSError:
            continue
        if value:
            return str(value)
    return ""


def gemini_key_source():
    """Where the key was found, for one friendly log line (never the key itself)."""
    if clean_key(os.environ.get(GEMINI_KEY_ENV)):
        return "process environment"
    if clean_key(_windows_persistent_env(GEMINI_KEY_ENV)):
        return "Windows user/machine environment"
    return ""


def gemini_key():
    """The Gemini API key, resolved so it is never missed.

    Checks ``os.environ["GEMINI_API_KEY"]`` first, then the persistent Windows
    environment, normalizing whitespace and stray quotes. Empty string = unset.
    """
    return (
        clean_key(os.environ.get(GEMINI_KEY_ENV))
        or clean_key(_windows_persistent_env(GEMINI_KEY_ENV))
    )


def gemini_model_name():
    """The Gemini model id this run will use (GEMINI_MODEL override, else the default)."""
    return clean_key(os.environ.get("GEMINI_MODEL")) or GEMINI_DEFAULT_MODEL


def gemini_text_part(payload):
    """Extract the plain text from a generateContent response body."""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    chunks = []
    for part in parts if isinstance(parts, list) else []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks)


def generate_text_gemini(prompt, attempts=JSON_TRIES):
    """Ask the Google Gemini API directly, bypassing the OpenCode free quota.

    Same contract as generate_text: returns the stripped reply text or raises Fail.
    The whole call is capped by GEMINI_BUDGET and every request by GEMINI_TIMEOUT,
    so a stalled or overloaded API can never hang a run.
    """
    global _GEMINI_ANSWERED_MODEL
    configured = clean_key(os.environ.get("GEMINI_MODEL"))
    ladder = list(GEMINI_MODEL_LADDER)
    if configured and configured not in ladder:
        ladder.insert(0, configured)   # try the override first, then the known-good ids
    deadline = time.monotonic() + GEMINI_BUDGET
    last_error = None
    for round_no in range(1, GEMINI_ROUNDS + 1):
        overloaded = False
        for index, model_id in enumerate(ladder):
            has_next = index < len(ladder) - 1
            if time.monotonic() >= deadline:
                break
            for attempt in range(1, attempts + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    break
                read_timeout = int(min(GEMINI_TIMEOUT[1], remaining))
                print(f"  asking {model_id} (up to {read_timeout}s)...")
                try:
                    response = requests.post(
                        GEMINI_ENDPOINT.format(model=model_id),
                        headers={"x-goog-api-key": gemini_key()},
                        json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
                        timeout=(min(GEMINI_TIMEOUT[0], remaining), read_timeout),
                    )
                except requests.Timeout:
                    last_error = f"{model_id} did not answer in time"
                    overloaded = True
                    print("  " + last_error
                          + ("; trying the next Gemini model" if has_next else ""))
                    break   # a stalled model -> next Gemini id, never wait again
                except requests.RequestException as exc:
                    raise Fail(
                        "could not reach the Google Gemini API "
                        f"({type(exc).__name__}); check the network connection and retry."
                    ) from exc
                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    text = gemini_text_part(payload)
                    if text.strip():
                        _GEMINI_ANSWERED_MODEL = model_id
                        return text.strip()
                    last_error = "Gemini returned an empty reply"
                    overloaded = True
                elif response.status_code == 404:
                    last_error = f"model {model_id} was not found"
                    if has_next:
                        print(f"  {model_id} is not available; trying the next Gemini model")
                    break  # unknown model id -> next entry in the ladder
                elif response.status_code in (401, 403):
                    raise Fail(
                        "Google rejected the GEMINI_API_KEY (HTTP "
                        f"{response.status_code}). Regenerate the key in AI Studio and update the "
                        "GEMINI_API_KEY environment variable."
                    )
                else:
                    message = (f"Gemini API error {response.status_code}: "
                               f"{str(response.text or '')[:300]}")
                    last_error = message
                    if response.status_code == 429:
                        overloaded = True
                        if attempt < attempts:
                            time.sleep(2 * attempt)
                            continue
                        break   # rate limited on this model -> next Gemini id
                    if response.status_code in (500, 502, 503):
                        overloaded = True
                        if has_next:
                            print(f"  {model_id} is busy (HTTP {response.status_code}); "
                                  "trying the next Gemini model")
                        break   # overloaded -> fail over now instead of waiting on it
                    raise Fail(message)
                if attempt < attempts:
                    time.sleep(1)
        # a fleet-wide spike is usually momentary: one bounded second pass
        if not overloaded or round_no >= GEMINI_ROUNDS:
            break
        if deadline - time.monotonic() <= GEMINI_RETRY_PAUSE + 5:
            break
        print(f"  every Gemini model is busy; retrying the ladder in {GEMINI_RETRY_PAUSE}s")
        time.sleep(GEMINI_RETRY_PAUSE)
    raise Fail(
        "Gemini analysis failed: "
        + (last_error or "no Gemini model answered within the time budget")
    )


def generate_text(prompt, model_ref, attempts=JSON_TRIES):
    """Ask the active model for text; Gemini when GEMINI_API_KEY is set, else OpenCode."""
    if gemini_key():
        return generate_text_gemini(prompt, attempts=attempts)
    url, password = opencode_service()
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                f"{url}/api/experimental/generate/text",
                auth=("opencode", password),
                json={"prompt": prompt, "model": model_ref},
                timeout=API_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise Fail(
                "could not reach the OpenCode background service. Open the OpenCode app "
                f"(or run 'opencode service start') and retry. ({type(exc).__name__})"
            ) from exc
        if response.status_code == 200:
            try:
                text = (response.json().get("data") or {}).get("text") or ""
            except ValueError:
                text = ""
            if text.strip():
                return text.strip()
            last_error = "the model returned an empty reply"
        elif response.status_code == 401:
            raise Fail(
                "OpenCode rejected the service password. Restart OpenCode so service.json is "
                "refreshed, then retry."
            )
        elif response.status_code == 404:
            free = list_free_models(url, password)
            hint = (" Available free models: " + ", ".join(free)) if free else ""
            raise Fail(f"model {model_ref.get('providerID')}/{model_ref.get('id')} was not found.{hint}")
        elif response.status_code == 422:
            raise Fail(provider_error_message(response.status_code, response.text))
        else:
            message = provider_error_message(response.status_code, response.text)
            if response.status_code in (429, 500, 502, 503) and "usage limit" not in message and attempt < attempts:
                last_error = message
                time.sleep(2 * attempt)
                continue
            raise Fail(message)
        if attempt < attempts:
            time.sleep(1)
    raise Fail(f"MiMo analysis failed after {attempts} attempts: {last_error}")


def ask_analysis(prompt, model_ref):
    """Ask MiMo for one window and require a valid JSON analysis back."""
    current = prompt
    last_problem = None
    for _ in range(JSON_TRIES):
        reply = generate_text(current, model_ref)
        try:
            return validate_analysis(extract_json(reply))
        except ValueError as exc:
            last_problem = str(exc)
            current = (
                prompt
                + "\n\nYour previous reply was invalid ("
                + last_problem
                + "). Reply again with ONLY the single JSON object described above."
            )
    raise Fail(f"MiMo never returned valid JSON: {last_problem}")


def fmt_model(model_ref):
    return f"{model_ref.get('providerID', '?')}/{model_ref.get('id', '?')}"


def active_model_label(model_ref):
    """The model that actually answers this run: Gemini wins whenever a key exists."""
    if gemini_key():
        return _GEMINI_ANSWERED_MODEL or gemini_model_name()
    return fmt_model(model_ref)


def ask_with_fallback(prompt, model_ref, fallbacks, call):
    """Run call(prompt, member); on a free-quota error advance to the next model.

    ``fallbacks`` may be "provider/model" strings or parsed refs; everything is
    normalized to a parsed ref before use. Returns (result, member_that_succeeded).
    When GEMINI_API_KEY is present the chain collapses to the primary entry, so
    no exhausted OpenCode model can ever be consulted.
    """
    primary = model_ref if isinstance(model_ref, dict) else parse_model_ref(model_ref)
    chain = [primary]
    if not gemini_key():
        for member in fallbacks or ():
            ref = member if isinstance(member, dict) else parse_model_ref(member)
            if fmt_model(ref) != fmt_model(primary):
                chain.append(ref)
    last_error = None
    previous = None
    for member in chain:
        if previous is not None:
            print(f"  free quota hit on {fmt_model(previous)} - switching to {fmt_model(member)}")
        try:
            return call(prompt, member), member
        except Fail as exc:
            if "usage limit" not in str(exc).lower():
                raise
            last_error = exc
            previous = member
    raise last_error or Fail("model analysis failed")


def parse_clip_list(text):
    """Pull clip objects out of a model reply (object, {"clips": [...]}, or bare array)."""
    text = str(text or "")
    decoder = json.JSONDecoder()
    for opener in ("[", "{"):
        idx = text.find(opener)
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except ValueError:
                idx = text.find(opener, idx + 1)
                continue
            items = None
            if opener == "[" and isinstance(obj, list):
                items = obj
            elif opener == "{" and isinstance(obj, dict):
                if isinstance(obj.get("clips"), list):
                    items = obj["clips"]
                elif "start" in obj:
                    items = [obj]
            if items is not None and any(
                isinstance(item, dict) and "start" in item for item in items
            ):
                return items
            idx = text.find(opener, idx + 1)
    raise ValueError("reply contained no clip list")


def ask_candidates(prompt, model_ref, count):
    """Ask for the clip list and return every structurally valid candidate (max count)."""
    current = prompt
    last_problem = None
    for _ in range(JSON_TRIES):
        reply = generate_text(current, model_ref)
        try:
            items = parse_clip_list(reply)
        except ValueError as exc:
            last_problem = str(exc)
        else:
            valid = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    valid.append(validate_analysis(item))
                except ValueError:
                    continue
            if valid:
                return valid[:count]
            last_problem = "reply contained no valid clip objects"
        current = (
            prompt
            + "\n\nYour previous reply was invalid ("
            + str(last_problem)
            + "). Reply again with ONLY the JSON described above."
        )
    raise Fail(f"the model never returned a valid clip list: {last_problem}")


def clips_overlap(a, b, gap=5):
    """Clips are distinct only when separated by at least `gap` seconds."""
    return not (a["end"] + gap <= b["start"] or b["end"] + gap <= a["start"])


def select_clips(candidates, count, floor=CLIP_SCORE_FLOOR, duration=None):
    """Pick 1..count genuinely strong, distinct moments - never padded.

    Quality gate: candidates below ``floor`` are dropped outright, so a video
    with one great moment ships one clip instead of filler. When ``duration``
    is known the timeline is split into ``count`` buckets and the best strong
    candidate is taken from each (beginning, middle, end); any remaining slots
    are filled by global score. Overlapping clips and duplicate titles are
    always removed. If nothing clears the floor, the single best candidate is
    returned so a run still ships something.
    """
    strong = [c for c in candidates if (c.get("score") or 0) >= floor]
    if not strong:
        best = max(candidates, key=lambda c: c.get("score") or 0, default=None)
        return [best] if best else []
    strong.sort(key=lambda c: c.get("score") or 0, reverse=True)
    picked = []
    titles = set()

    def acceptable(cand):
        title_key = str(cand.get("title") or "").strip().lower()
        if title_key and title_key in titles:
            return False        # duplicate metadata -> not a distinct moment
        return not any(clips_overlap(cand, p) for p in picked)

    def take(cand):
        picked.append(cand)
        title_key = str(cand.get("title") or "").strip().lower()
        if title_key:
            titles.add(title_key)

    if duration and duration > 0 and count > 1:
        span = float(duration) / count
        for bucket in range(count):
            if len(picked) >= count:
                break
            low, high = bucket * span, (bucket + 1) * span
            for cand in strong:
                if low <= cand["start"] < high and acceptable(cand):
                    take(cand)
                    break

    for cand in strong:
        if len(picked) >= count:
            break
        if acceptable(cand):
            take(cand)

    picked.sort(key=lambda c: c.get("score") or 0, reverse=True)   # best first, as promised
    return picked[:count]


# ---------- yt-dlp: metadata + transcript + section ----------
def ytdlp_cmd(*args):
    """yt-dlp needs a JavaScript runtime for YouTube; fall back to node/bun when deno is missing."""
    cmd = [sys.executable, "-m", "yt_dlp", "--socket-timeout", "30"]
    if not shutil.which("deno"):
        for runtime in ("node", "bun"):
            if shutil.which(runtime):
                cmd += ["--js-runtimes", runtime]
                break
    if not shutil.which("ffmpeg"):
        try:  # section cutting needs ffmpeg; point yt-dlp at the copy we found
            cmd += ["--ffmpeg-location", str(Path(find_ffmpeg()).parent)]
        except Fail:
            pass
    return cmd + list(args)


def run(cmd, timeout=None, **kwargs):
    """Run a subprocess safely: no stdin, optional hard timeout, never a zombie.

    ``timeout`` stops a stalled download or encode instead of hanging the
    terminal; the child is killed and a user-facing Fail is raised.
    """
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    try:
        return subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace",
            timeout=timeout, **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        name = Path(str(cmd[0])).name if cmd else "command"
        raise Fail(
            f"{name} ran longer than {int(timeout)}s and was stopped so the "
            "computer stays responsive. Re-run when the connection is faster, "
            "or raise the timeout."
        ) from exc


VIDEO_URL_RE = re.compile(
    r"youtube\.com/(?:watch\?|shorts/|live/|embed/|v/)|youtu\.be/", re.I
)


def resolve_video_url(url):
    """Accept a video, Shorts, livestream, or channel link.

    Direct video links pass through untouched. A channel/handle/custom link is
    resolved to its newest upload so a single pasted URL is all a run needs.
    """
    if VIDEO_URL_RE.search(url):
        return url
    base = url.split("?")[0].rstrip("/")
    for target in (f"{base}/videos", base):
        result = run(
            ytdlp_cmd("--flat-playlist", "--dump-single-json", "--playlist-end", "5",
                      "--no-warnings", target),
            timeout=YTDLP_TIMEOUT,
        )
        if result.returncode:
            continue
        try:
            data = json.loads(result.stdout)
        except ValueError:
            continue
        for entry in data.get("entries") or []:
            video_id = (entry or {}).get("id")
            if video_id:
                resolved = f"https://www.youtube.com/watch?v={video_id}"
                print(f"      channel link resolved to newest upload: {resolved}")
                return resolved
    raise Fail(
        "that link is not a video and no upload could be resolved from it. "
        "Pass a video, Shorts, livestream, or channel URL."
    )


def stream_info(url):
    result = run(
        ytdlp_cmd("--dump-single-json", "--skip-download", "--no-playlist", "--no-warnings", url),
        timeout=YTDLP_TIMEOUT,
    )
    if result.returncode:
        raise Fail("yt-dlp could not read that URL:\n" + (result.stderr or "")[-400:])
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise Fail("yt-dlp returned unreadable metadata") from exc


def parse_srt(content):
    cues = []
    for block in re.split(r"\n\s*\n", content.replace("\r\n", "\n").strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        match_i = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if match_i is None:
            continue
        stamp = lines[match_i]
        left, _, right = stamp.partition("-->")
        try:
            start = parse_timestamp(left.strip().replace(",", "."))
            end = parse_timestamp(right.split()[0].strip().replace(",", "."))
        except ValueError:
            continue
        text = " ".join(l.strip() for l in lines[match_i + 1:]).strip()
        if text and end > start:
            cues.append({"start": start, "end": end, "text": text})
    cues.sort(key=lambda c: c["start"])
    return cues


def fetch_transcript(url, langs, workdir):
    """Download subtitles only: --skip-download keeps video bandwidth at zero."""
    result = run(
        ytdlp_cmd(
            "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", langs, "--sub-format", "srt/best", "--convert-subs", "srt",
            "--no-playlist", "--no-warnings", "-o", "subs.%(ext)s", url,
        ),
        cwd=str(workdir),
        timeout=YTDLP_TIMEOUT,
    )
    files = sorted(Path(workdir).glob("subs*.srt"))
    if not files:
        detail = (result.stderr or "").strip().splitlines()
        hint = " Try --sub-langs all if the video uses another language."
        raise Fail(
            "no subtitles or auto-captions were found for that video." + hint
            + (("\n" + detail[-1]) if detail else "")
        )
    files.sort(key=lambda p: (".auto" in p.name, p.name))  # prefer manual captions
    cues = parse_srt(files[0].read_text(encoding="utf-8", errors="replace"))
    cues = [c for c in cues if c["text"]]
    if not cues:
        raise Fail("the subtitle file was empty or unreadable")
    return cues, files[0].name


def download_section(url, start, end, workdir, info, attempts=3):
    """Fetch only [start, end) of the source.

    Section downloads ride on HTTP range requests, so a dropped connection
    mid-transfer is a routine, retryable failure rather than a bad URL.
    """
    if info.get("is_live"):
        print(
            "  note: this source is a live stream; the cut only works if YouTube still serves "
            "that part of the stream (DVR) and the captions covered it."
        )
    fmt = "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b"
    cmd = ytdlp_cmd(
        "--download-sections", section_range(start, end), "--force-keyframes-at-cuts",
        "--no-playlist", "--no-warnings", "-f", fmt, "-o", "section.%(ext)s", url,
    )
    result = None
    for attempt in range(1, attempts + 1):
        result = run(cmd, cwd=str(workdir), timeout=SECTION_TIMEOUT)
        files = [
            p for p in sorted(Path(workdir).glob("section.*"))
            if p.suffix not in {".part", ".json"} and p.is_file()
        ]
        if files:
            return files[0]
        if attempt < attempts:
            for stale in Path(workdir).glob("section.*"):  # clear partials
                stale.unlink(missing_ok=True)
            print(
                f"  section download produced no file "
                f"(attempt {attempt}/{attempts}); retrying..."
            )
    raise Fail(
        "could not download that timestamp range:\n" + (result.stderr or "")[-500:]
        + ("\nThe range may be outside a live stream's DVR window." if info.get("is_live") else "")
    )


# ---------- FFmpeg: 9:16 crop + burned captions ----------
def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        Path(os.environ["LOCALAPPDATA"]) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe" if os.environ.get("LOCALAPPDATA") else None,
        Path.home() / "scoop" / "shims" / "ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
        HERE / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for path in candidates:
        if path and path.is_file():
            return str(path)
    raise Fail(FFMPEG_HINT)


def check_ass_filter(ffmpeg):
    result = run([ffmpeg, "-hide_banner", "-filters"], timeout=60)
    if result.returncode == 0 and re.search(r"\bass\b", result.stdout) is None:
        raise Fail(
            "this FFmpeg build has no 'ass' subtitle filter (libass missing). "
            "Install a full build such as https://www.gyan.dev/ffmpeg/builds/ (essentials)."
        )


def probe_has_audio(ffmpeg, path):
    result = run([ffmpeg, "-hide_banner", "-i", str(path)], timeout=60)
    return "Audio:" in (result.stderr or "")


def ass_time(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def write_ass(path, cues, start, end, hook=""):
    """Burned captions: white Arial Black on a thick black outline, high contrast.

    ``hook`` is the AI-suggested on-screen text hook burned top-center for the
    whole clip (plain ASCII, already normalized by enforce_analysis).
    """
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Clip,Arial Black,76,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,"
        "-1,0,0,0,100,100,0,0,1,6,2,2,60,60,320,1\n"
        "Style: Hook,Arial Black,64,&H0000FFFF,&H00FFFFFF,&H00000000,&H96000000,"
        "-1,0,0,0,100,100,0,0,1,7,3,8,60,60,150,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    events = []
    for cue in cues:
        rel_start = max(0.0, cue["start"] - start)
        rel_end = min(float(end - start), cue["end"] - start)
        if rel_end <= rel_start:
            continue
        text = " ".join(str(cue["text"]).split())
        text = text.replace("\\", "").replace("{", "(").replace("}", ")")
        if not text:
            continue
        events.append(f"Dialogue: 0,{ass_time(rel_start)},{ass_time(rel_end)},Clip,,0,0,0,,{text}")
    hook = " ".join(str(hook or "").split())
    if hook:
        hook = hook.replace("\\", "").replace("{", "(").replace("}", ")")
        events.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(end - start)},Hook,,0,0,0,,{hook}"
        )
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")


def build_render_cmd(ffmpeg, section_path, out_name, filter_graph, maps, length,
                     has_audio, threads=0, preset=X264_PRESET, crf=X264_CRF):
    """Assemble the x264 command for weak CPUs.

    - ``-nostdin`` + DEVNULL stdin: FFmpeg can never wait on keyboard input
    - ``preset`` (default veryfast) + cheap ``bicubic`` scaling: low CPU load
    - ``-threads``: 0 = auto, or cap it (e.g. 2) so the PC stays usable
    - ``+faststart``: YouTube-friendly progressive MP4
    """
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(section_path),
        "-filter_complex", filter_graph,
        *maps,
        "-t", str(length),
        "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
        "-threads", str(max(0, int(threads))),
        "-pix_fmt", "yuv420p",
        *(["-c:a", "aac", "-b:a", "160k"] if has_audio else ["-an"]),
        "-movflags", "+faststart", "-avoid_negative_ts", "make_zero",
        str(out_name),
    ]


def build_video_filter(ass_name):
    """Cheap deterministic 9:16 chain: crop -> bicubic scale -> burn captions.

    ``bicubic`` instead of ``lanczos`` keeps old CPUs responsive during the
    scale step with no visible difference at 1080x1920.
    """
    return (
        "crop='trunc(min(iw,ih*9/16)/2)*2':'trunc(min(ih,iw*16/9)/2)*2',"
        f"scale=1080:1920:flags={SCALER_FLAGS},setsar=1,setpts=PTS-STARTPTS,"
        f"ass={ass_name}[v]"
    )


def render_vertical(ffmpeg, section_path, cues, start, end, workdir, hook="",
                    threads=0, preset=X264_PRESET, crf=X264_CRF):
    """Center-crop to 9:16, scale to 1080x1920, burn captions + hook, keep the source audio."""
    length = end - start
    ass_name = "captions.ass"
    write_ass(Path(workdir) / ass_name, cues, start, end, hook=hook)
    video_chain = build_video_filter(ass_name)
    has_audio = probe_has_audio(ffmpeg, section_path)
    if has_audio:
        filter_graph = f"[0:v]{video_chain};[0:a]asetpts=PTS-STARTPTS[a]"
        maps = ["-map", "[v]", "-map", "[a]"]
    else:
        filter_graph = f"[0:v]{video_chain}"
        maps = ["-map", "[v]"]
    out_path = Path(workdir) / "final.mp4"
    cmd = build_render_cmd(
        ffmpeg, section_path, out_path.name, filter_graph, maps, length,
        has_audio, threads=threads, preset=preset, crf=crf,
    )
    result = run(cmd, cwd=str(workdir), timeout=RENDER_TIMEOUT)
    (Path(workdir) / ass_name).unlink(missing_ok=True)
    if result.returncode or not out_path.is_file() or out_path.stat().st_size == 0:
        raise Fail("FFmpeg failed to render the vertical clip:\n" + (result.stderr or "")[-600:])
    return out_path


def extract_cover(ffmpeg, video_path, peak, start, out_path, length=None):
    """Grab the peak-payoff frame as the cover/thumbnail reference.

    Tries the AI-flagged peak second first, then the clip midpoint and the
    first second, so a broken or blank frame never leaves a clip without a
    cover. Failures return None instead of killing the run (a cover is a
    reference, not the deliverable).
    """
    offsets = [max(0.0, float(peak or 0) - float(start or 0))]
    if length:
        offsets.append(max(0.0, float(length) / 2.0))
    offsets.append(0.0)
    seen = set()
    for rel in offsets:
        rel = round(rel, 2)
        if rel in seen:
            continue
        seen.add(rel)
        result = run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", f"{rel:.2f}", "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_path),
        ], timeout=COVER_TIMEOUT)
        if result.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 1024:
            return out_path
        out_path.unlink(missing_ok=True)
    return None


# ---------- storage + YouTube upload ----------
def save_outputs(analysis, source_info, final_path, outdir, model_ref, transcript_name,
                 cover_path=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slug = sanitize_filename(analysis["title"]) or str(source_info.get("id") or "clip")
    stem = (
        f"{slug}_{source_info.get('id', 'video')}"
        f"_{fmt_ts(analysis['start']).replace(':', '')}-{fmt_ts(analysis['end']).replace(':', '')}"
    )
    video_path = outdir / f"{stem}.mp4"
    counter = 2
    while video_path.exists():
        video_path = outdir / f"{stem}_{counter}.mp4"
        counter += 1
    shutil.move(str(final_path), str(video_path))
    cover_out = None
    if cover_path and Path(cover_path).is_file():
        cover_out = video_path.with_suffix(".jpg")
        shutil.move(str(cover_path), str(cover_out))
    metadata = {
        "source_url": source_info.get("webpage_url") or source_info.get("original_url"),
        "source_id": source_info.get("id"),
        "source_title": source_info.get("title"),
        "start": fmt_ts(analysis["start"]),
        "end": fmt_ts(analysis["end"]),
        "peak": fmt_ts(analysis["peak"]) if analysis.get("peak") is not None else None,
        "length_seconds": analysis["end"] - analysis["start"],
        "title": analysis["title"],
        "description": analysis["description"],
        "hashtags": analysis["hashtags"],
        "hook": analysis.get("hook", ""),
        "cover": cover_out.name if cover_out else None,
        "tags": analysis["tags"],
        "reason": analysis["reason"],
        "score": analysis["score"],
        "model": analysis.get("_model") or f"{model_ref['providerID']}/{model_ref['id']}",
        "transcript_file": transcript_name,
        "format": "1080x1920 9:16",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "instagram": "manual upload from this folder",
        "youtube": None,
    }
    meta_path = video_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return video_path, meta_path, metadata


def youtube_body(metadata, privacy):
    title = ensure_shorts_title(metadata["title"])
    description = str(metadata["description"])[:4600]
    missing = [tag for tag in HASHTAGS if tag not in description]
    if missing:  # the slice must never clip the required hashtag line
        description = (description + "\n\n" + " ".join(missing)).strip()
    tags = [t for t in metadata["tags"] if isinstance(t, str)]
    return {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "20",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }


def upload_short(video_path, metadata, privacy, client_secrets):
    """Upload to the authorized YouTube channel with the AI-generated SEO metadata."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        raise Fail(
            "YouTube upload needs google-api-python-client and google-auth-oauthlib: "
            "pip install google-api-python-client google-auth-oauthlib"
        ) from exc
    client_secrets = Path(client_secrets)
    if not client_secrets.is_file():
        raise Fail(
            f"{client_secrets} not found. Download OAuth client secrets (Desktop app) from "
            "https://console.cloud.google.com/apis/credentials with the YouTube Data API v3 enabled."
        )
    creds = None
    if YOUTUBE_TOKEN.is_file():
        try:
            creds = Credentials.from_authorized_user_file(str(YOUTUBE_TOKEN), YOUTUBE_SCOPES)
        except ValueError:
            creds = None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if not sys.stdin.isatty():
            # unattended runs must never hang on a browser nobody can see
            raise Fail(
                "the YouTube token is missing or expired and this run is non-interactive. "
                "Run the same command once from an interactive terminal to authorize; "
                f"it writes {YOUTUBE_TOKEN.name} and later runs reuse it."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), YOUTUBE_SCOPES)
        print("  opening the browser for YouTube consent (one-time per token)...")
        creds = flow.run_local_server(port=0, open_browser=True)
    YOUTUBE_TOKEN.write_text(creds.to_json(), encoding="utf-8")
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = youtube_body(metadata, privacy)
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    attempt = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"  upload {int(status.progress() * 100)}%")
        except HttpError as exc:
            code = getattr(exc.resp, "status", None)
            attempt += 1
            if code == 401 and creds and attempt <= 2:
                try:
                    creds.refresh(Request())
                    YOUTUBE_TOKEN.write_text(creds.to_json(), encoding="utf-8")
                    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
                    continue
                except Exception:
                    pass  # token is unusable; fall through to the error below
            if code in (429, 500, 502, 503, 504) and attempt <= 5:
                time.sleep(min(60, 5 * attempt))
                continue
            raise Fail(f"YouTube upload failed ({code}): {str(exc)[:300]}") from exc
    return response.get("id")


# ---------- pipeline ----------
def analyze(cues, model_ref, info, fallbacks=()):
    duration = info.get("duration") or (cues[-1]["end"] if cues else None)
    windows = chunk_cues(cues)
    print(f"  transcript: {len(cues)} cues, {len(windows)} model call(s)")
    best = None
    for index, window in enumerate(windows, 1):
        if len(windows) > 1:
            print(f"  analyzing part {index}/{len(windows)}...")
        prompt = build_prompt(
            transcript_block(window), index, len(windows), info.get("title"), duration,
        )
        raw, member = ask_with_fallback(prompt, model_ref, fallbacks, ask_analysis)
        analysis = enforce_analysis(raw, duration)
        analysis["_model"] = active_model_label(member)
        if best is None or analysis["score"] > best["score"]:
            best = analysis
    return best


def analyze_multi(cues, model_ref, info, count, fallbacks=()):
    """Find up to `count` distinct high-retention moments across the transcript."""
    duration = info.get("duration") or (cues[-1]["end"] if cues else None)
    windows = chunk_cues(cues)
    print(
        f"  transcript: {len(cues)} cues, {len(windows)} model call(s), "
        f"asking for up to {count} distinct clips"
    )
    candidates = []
    for index, window in enumerate(windows, 1):
        if len(windows) > 1:
            print(f"  analyzing part {index}/{len(windows)}...")
        prompt = build_prompt(
            transcript_block(window), index, len(windows), info.get("title"), duration,
            count=count,
        )
        items, member = ask_with_fallback(
            prompt, model_ref, fallbacks,
            lambda text_prompt, member_ref: ask_candidates(text_prompt, member_ref, count),
        )
        for item in items:
            clip = enforce_analysis(item, duration)
            clip["_model"] = active_model_label(member)
            candidates.append(clip)
    clips = select_clips(candidates, count, duration=duration)
    if len(clips) < len(candidates):
        print(f"  kept {len(clips)} of {len(candidates)} returned moment(s) after "
              "quality/overlap/spread filtering")
    if not clips:
        raise Fail("the model found no usable moments in this transcript")
    return clips


def build_parser():
    """CLI definition, separated from main() so defaults are testable on their own."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", nargs="?", help="YouTube live stream or long-form video URL")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenCode model ref, used only when GEMINI_API_KEY is unset "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--clips", type=int, default=CLIPS_DEFAULT, choices=range(1, CLIPS_MAX + 1),
                        metavar="N",
                        help=f"maximum distinct viral moments to extract, 1-{CLIPS_MAX} "
                             f"(default: {CLIPS_DEFAULT}); only genuinely strong, well-spread "
                             "moments ship, so 1..N clips come back")
    parser.add_argument("--threads", type=int, default=0, metavar="N",
                        help="x264 encoder threads; 0 = auto, set 2 so the PC stays usable")
    parser.add_argument("--x264-preset", default=X264_PRESET,
                        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
                        help=f"x264 speed/quality preset (default: {X264_PRESET})")
    parser.add_argument("--upload-gap", type=int, default=UPLOAD_GAP_DEFAULT, metavar="SECONDS",
                        help=f"pause between sequential uploads (default: {UPLOAD_GAP_DEFAULT})")
    parser.add_argument("--no-model-fallback", action="store_true",
                        help="disable automatic model switching on quota errors "
                             f"(chain: {', '.join(FALLBACK_MODELS)})")
    parser.add_argument("--sub-langs", default="en.*,en",
                        help="subtitle languages for the transcript, yt-dlp syntax (default: en.*,en; try all)")
    parser.add_argument("--output-dir", default=str(HERE / DEFAULT_OUTPUT_DIR),
                        help=f"folder for the manual Instagram copy (default: {DEFAULT_OUTPUT_DIR}/)")
    parser.add_argument("--client-secrets", default=str(CLIENT_SECRETS),
                        help="YouTube OAuth client secrets JSON")
    parser.add_argument("--privacy", choices=("public", "unlisted", "private"), default="public",
                        help="YouTube visibility (default: public)")
    parser.add_argument("--skip-upload", action="store_true",
                        help="save the clip and metadata only; never touch YouTube")
    parser.add_argument("--dry-run", action="store_true",
                        help="stop after the model analysis and print the plan")
    parser.add_argument("--keep-work", action="store_true",
                        help="keep the temporary download/render folder")
    return parser


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args()

    # fail fast on missing tools before asking for anything
    model_ref = parse_model_ref(args.model)
    ffmpeg = find_ffmpeg()
    check_ass_filter(ffmpeg)
    if gemini_key():
        # Gemini key present -> the OpenCode service and its free quota are irrelevant
        print(f"      AI: Google Gemini API ({gemini_model_name()}); "
              f"key found in {gemini_key_source()}; OpenCode quota is not used")
    else:
        opencode_service()  # fail before any download if OpenCode is not installed/running

    if not args.url:
        if not sys.stdin.isatty():
            sys.exit(
                "error: a YouTube URL is required when running non-interactively "
                '(e.g. python viral_shorts.py "https://www.youtube.com/watch?v=...")'
            )
        try:
            args.url = input("YouTube URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit("\nstopped")
    if not args.url:
        sys.exit("a YouTube URL is required")

    workdir = Path(tempfile.mkdtemp(prefix="viral_shorts_"))
    try:
        args.url = resolve_video_url(args.url)
        print(f"[1/6] Reading video info: {args.url}")
        info = stream_info(args.url)
        title = info.get("title") or "untitled video"
        duration = info.get("duration")
        print(f"      {title}" + (f" ({fmt_ts(duration)})" if duration else ""))

        print("[2/6] Downloading subtitles only (no video data)...")
        cues, sub_name = fetch_transcript(args.url, args.sub_langs, workdir)
        print(f"      {len(cues)} caption cues from {sub_name}")

        if gemini_key():
            # one provider end to end: no OpenCode fallback chain to consult
            fallbacks = []
            ai_note = f"Google Gemini {gemini_model_name()}"
        else:
            fallbacks = [] if args.no_model_fallback else list(FALLBACK_MODELS)
            chain = ", ".join(fallbacks) if fallbacks else "off"
            ai_note = f"{args.model}; fallbacks: {chain}"
        if args.clips > 1:
            print(f"[3/6] Asking for up to {args.clips} distinct {SEGMENT_MIN}-{SEGMENT_MAX}s "
                  f"viral moments ({ai_note})...")
            clips = analyze_multi(cues, model_ref, info, args.clips, fallbacks=fallbacks)
        else:
            print(f"[3/6] Asking for the best {SEGMENT_MIN}-{SEGMENT_MAX}s moment "
                  f"({ai_note})...")
            clips = [analyze(cues, model_ref, info, fallbacks=fallbacks)]
        for i, clip in enumerate(clips, 1):
            label = f"      clip {i}/{len(clips)}: " if len(clips) > 1 else "      "
            print(f"{label}{fmt_ts(clip['start'])} - {fmt_ts(clip['end'])} "
                  f"(score {clip['score']:.0f}, model {clip.get('_model', args.model)})")
            print(f"{label}title: {clip['title']}")
            if clip.get("hook"):
                print(f"{label}hook: {clip['hook']}")

        if args.dry_run:
            print("\nDry run complete. Plan:")
            print(json.dumps(clips if len(clips) > 1 else clips[0],
                             ensure_ascii=False, indent=2))
            return 0

        saved = []
        total = len(clips)
        for i, analysis in enumerate(clips, 1):
            tag = f"clip {i}/{total}: " if total > 1 else ""
            print(f"[4/6] {tag}cutting "
                  f"{section_range(analysis['start'], analysis['end']).lstrip('*')} ...")
            section = download_section(args.url, analysis["start"], analysis["end"], workdir, info)

            print(f"[5/6] {tag}rendering 1080x1920 9:16 with captions + hook "
                  f"({args.x264_preset})...")
            final = render_vertical(
                ffmpeg, section, cues, analysis["start"], analysis["end"], workdir,
                hook=analysis.get("hook", ""),
                threads=args.threads, preset=args.x264_preset,
            )
            cover = extract_cover(
                ffmpeg, final, analysis.get("peak"), analysis["start"],
                workdir / f"cover_{i}.jpg",
                length=analysis["end"] - analysis["start"],
            )

            print(f"[6/6] {tag}saving to {args.output_dir} ...")
            video_path, meta_path, metadata = save_outputs(
                analysis, info, final, args.output_dir, model_ref, sub_name,
                cover_path=cover,
            )
            print(f"      Instagram (manual): {video_path}")
            if metadata.get("cover"):
                print(f"      cover frame: {Path(args.output_dir) / metadata['cover']}")
            saved.append((video_path, meta_path, metadata))

        if args.skip_upload:
            print("      YouTube: skipped (--skip-upload)")
            return 0

        failures = []
        gap = max(0, args.upload_gap)
        for i, (video_path, meta_path, metadata) in enumerate(saved, 1):
            if len(saved) > 1:
                print(f"      Uploading clip {i}/{len(saved)} ({video_path.name})...")
            else:
                print("      Uploading to YouTube as a Short...")
            try:
                video_id = upload_short(video_path, metadata, args.privacy, args.client_secrets)
            except Fail as exc:
                if len(saved) == 1:
                    raise
                failures.append((video_path.name, str(exc)))
                print(f"      upload FAILED: {exc}")
            else:
                metadata["youtube"] = {
                    "id": video_id,
                    "url": f"https://www.youtube.com/shorts/{video_id}",
                    "privacy": args.privacy,
                    "uploaded": datetime.now().astimezone().isoformat(timespec="seconds"),
                }
                with meta_path.open("w", encoding="utf-8") as file:
                    json.dump(metadata, file, ensure_ascii=False, indent=2)
                print(f"      YouTube: {metadata['youtube']['url']}")
            if i < len(saved) and gap:
                print(f"      waiting {gap}s before the next upload...")
                time.sleep(gap)
        if failures:
            print(f"\n{len(failures)} of {len(saved)} uploads failed:")
            for name, error in failures:
                print(f"  - {name}: {error}")
            return 1
        print(f"\nDone. {len(saved)} clip(s) published to YouTube and saved in {args.output_dir}.")
        print("Review them, then upload to Instagram Reels from the folder above.")
        return 0
    finally:
        if args.keep_work:
            print(f"work folder kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as exc:
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nstopped")
