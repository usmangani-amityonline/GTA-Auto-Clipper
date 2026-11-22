# GTA Auto Clipper

Find high-energy moments in live gameplay and render full-screen 9:16 clips with pop-up captions: each word pops in yellow the moment it is spoken. The clipper runs on your computer. A separate Zernio command can schedule the finished clip to your connected YouTube and Instagram accounts.

The software is free and MIT-licensed. The optional AI workflow guide and prompt pack are a separate digital product.

## What it does

- Searches current YouTube live results for a phrase such as `GTA 6` and prints machine-readable JSON for AI agents.
- Requires an explicit visual confirmation before live capture, so a discussion stream or facecam-only source is not silently treated as gameplay.
- Records a live source in rolling chunks. By default it ranks moments by live-chat reactions, then falls back to audio and speech if chat is unavailable or has no messages for a chunk. Combined chat, audio, and speech scoring is also available.
- Starts clips a few seconds before the detected highlight (6 seconds by default).
- Captions English speech in English and Hindi speech in romanized Hinglish. Other or uncertain languages, including Hindi the recognizer returns in an unsupported script, have no captions. Captions are short one-line phrases (up to 3 words) at one fixed spot: each word pops in yellow when it is spoken, earlier words turn white, and the next phrase replaces the last one in the same place (never stacked). Source audio stays unchanged.
- Places captions in the lower third of the visible gameplay panel instead of using one fixed position near the bottom of the full vertical canvas.
- Fills the whole 9:16 screen with gameplay by default (`--layout fill`), centered on the player. It samples frames for facecams, shifts the crop away from a persistent side facecam, and in stacked portrait streams cuts out the facecam and fills the screen with the gameplay part. `--layout fit` keeps the old look: the full source in the middle over a blurred background.
- Resets audio and video timestamps before burning the captions.
- Masks swear words and sexual words in the burned-in captions (for example `s***`) so platforms do not limit the clip; the original audio is unchanged.
- Saves MP4 and JSON metadata in `output/`. It never posts as part of clipping.
- Schedules reviewed clips through Zernio by default. Optional unattended scheduling is available after account setup and with a standing source-rights policy; dry-run is the default.

## Install

Requirements: Python 3.10+, FFmpeg in PATH, Node.js or Deno (yt-dlp needs a JavaScript runtime to read YouTube; the clipper uses Node automatically when Deno is missing), OpenCV for `--crop` face detection, and a stable connection for first-run speech-model downloads.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For live-stream search and optional YouTube live-chat scoring, set a YouTube Data API key in the terminal. Keep keys out of prompts, source files, and GitHub:

```powershell
$env:YOUTUBE_API_KEY = "YOUR_YOUTUBE_DATA_API_KEY"
python find_streams.py "GTA 6" --max 25 --json
```

The YouTube API returns live search candidates in `viewCount` order. It is not a dependable ranking of current live viewers; open a candidate and confirm the stream before use.

## Make clips

Open the candidate stream and verify that actual GTA gameplay is visible in the video. A GTA title, thumbnail, or talking head is not enough. Then start capture and keep it running until you press Ctrl+C. The default is chat-first. If chat is unavailable or has no messages for a chunk, the clipper automatically uses audio and speech instead. The clipper writes each selected highlight to `output/`:

```powershell
python clip.py "https://www.youtube.com/watch?v=VIDEO_ID" --gameplay-confirmed --max-seconds 0 --clip-len 30 --lead-in 6 --top 2
```

Important options:

| Option | Default | Purpose |
| --- | ---: | --- |
| `--max-seconds` | `0` | Capture duration; `0` runs until stopped |
| `--chunk` | `90` | Analysis chunk size in seconds |
| `--clip-len` | `30` | Rendered highlight length |
| `--lead-in` | `6` | Seconds before the detected moment |
| `--top` | `1` | Highlights kept from each chunk |
| `--gameplay-confirmed` | required for live capture | Confirm you inspected the stream and saw actual gameplay before recording |
| `--min-score` | `2.5` | Minimum moment score |
| `--layout` | `fill` | `fill`: gameplay covers the full 9:16 screen, facecams cropped out. `fit`: full source in the middle over a blurred background |
| `--crop` | off | Old flag, same as `--layout fill` |
| `--moment-source` | `chat` | Choose `chat`, `combined`, or `audio` for highlight selection |
| `--spoken-language` | `auto` | `auto` detects the language and locks to Hindi (Hinglish captions) once a stream is clearly Hindi. `hi` / `en` force it for the whole stream |
| `--no-chat` | off | Skip live-chat collection and use audio and speech only |
| `--output-dir` | `output/` | Custom destination folder |

Examples: `--moment-source chat` uses chat reactions first; `--moment-source combined` scores chat, audio, and speech together; `--moment-source audio` uses the original audio and speech method. Chat mode falls back to audio and speech when it has no chat data.

Test a local video first. Test output goes to `test_out/` and cannot be published:

```powershell
python clip.py --file "C:\videos\gameplay.mp4" --test --clip-len 30
```

## Schedule to YouTube + Instagram

Connect your YouTube account and an Instagram Creator or Business account to Zernio. Set the key and account IDs as private environment variables:

```powershell
$env:ZERNIO_API_KEY = "YOUR_ZERNIO_API_KEY"
$env:ZERNIO_YOUTUBE_ACCOUNT_ID = "YOUR_YOUTUBE_ACCOUNT_ID"
$env:ZERNIO_INSTAGRAM_ACCOUNT_ID = "YOUR_INSTAGRAM_ACCOUNT_ID"
```

Review the clip and its JSON metadata, then mark the metadata `"approved": true`. Preview the schedule first:

```powershell
python zernio_publish.py "output\gta-CLIP-ID" --scheduled-for "2026-11-18T12:00:00+05:30"
```

The dry run sends nothing. To schedule the post to both accounts, add `--publish`:

```powershell
python zernio_publish.py "output\gta-CLIP-ID" --scheduled-for "2026-11-18T12:00:00+05:30" --publish
```

For unattended posting, first make sure the AI agent is restricted to sources you have the right to use and that you have tested its output. Then set `GTA_UNATTENDED_POSTS=1` in the machine's private environment and run with `--publish --unattended`. A scheduled post already accepted by Zernio can publish while your computer is asleep. Discovery and clip rendering still require an awake, connected runner.

`publish.py` is the same command as `zernio_publish.py`, but it also accepts the generic variable names `PUBLISH_API_KEY`, `PUBLISH_YOUTUBE_ACCOUNT_ID` and `PUBLISH_INSTAGRAM_ACCOUNT_ID`, so either spelling works: `python publish.py "output\gta-CLIP-ID" --publish --unattended --scheduled-for "FUTURE_ISO_TIME"`. To post to many channels with an equal share, use [GTA Auto Sender](https://github.com/AnIntellectualBeing/GTA-Auto-Sender).

The publisher stores the Zernio post ID and an idempotency key in the metadata. If an API call fails, check the Zernio dashboard before retrying.

## AI viral Shorts (MiMo + OpenCode)

`viral_shorts.py` turns one URL into a finished vertical Short in a single command — paste a video, Shorts, livestream, or even a channel link (channel links resolve to the newest upload). It downloads only the subtitles (no video bytes for analysis), sends the transcript to the active LLM (MiMo by default, any OpenCode model via `--model`, e.g. GLM) with a strict pro-editor prompt — high-energy moments, shock value, screams, fails, climax hype, opening immediately on the hook — for the single best 35-50 second window, cuts only that range with `yt-dlp --download-sections`, crops it to 1080x1920 9:16 with FFmpeg, burns in high-contrast captions **plus the AI-suggested on-screen text hook** (top-center, yellow on black outline), then:

- saves the clip and its JSON metadata (including the hook text) in `instagram_manual_ready/` for your manual, safe Instagram Reels upload, and
- uploads the same file to your YouTube channel as a Short with the AI-generated high-SEO metadata: a curiosity-driven title under 60 characters ending in `#Shorts`, a keyword-rich description carrying exactly `#Shorts #Gaming #GTA6 #Viral #Trending #GamerMoments`, and 8-15 tags (`client_secrets.json` + YouTube Data API v3).

Requirements on top of the base install: FFmpeg with the `ass` filter (a full build such as gyan.dev essentials), the OpenCode app running in the background (it registers `~/.local/state/opencode/service.json`, which the script reads for API access), and YouTube upload credentials: enable YouTube Data API v3 in Google Cloud, create an OAuth Desktop client, and save it as `client_secrets.json` in this folder. The first upload opens the browser once for consent; the token is cached in `youtube_token.json` (both files are gitignored).

```powershell
# full run: analyze, cut, render, save to instagram_manual_ready/, upload a Short
python viral_shorts.py "https://www.youtube.com/watch?v=VIDEO_ID"

# channel or handle link: resolves to the newest upload, then runs as above
python viral_shorts.py "https://www.youtube.com/@SomeChannel"

# analysis only - prints the chosen segment, title, description, hashtags, tags
python viral_shorts.py "URL" --dry-run

# save the finished clip locally without touching YouTube
python viral_shorts.py "URL" --skip-upload

# transcripts in other languages, e.g. Hindi + English
python viral_shorts.py "URL" --sub-langs "hi.*,hi,en.*,en"
```

Options: `--clips N` (maximum 1-5 distinct moments per run; only genuinely strong, well-spread moments ship, so 1..N clips come back — no padding with weak or overlapping filler), `--upload-gap SECONDS` (default 90 between sequential uploads), `--no-model-fallback`, `--model` (default `opencode/mimo-v2.6-flash-free`; on a quota error the run automatically switches to `opencode/nemotron-3.5-super-vl-free`, then `opencode/deepseek-v4.1-flash-free`), `--sub-langs` (default `en.*,en`; use `all` when auto-captions are in another language), `--output-dir` (default `instagram_manual_ready/`), `--privacy public|unlisted|private` (default `public`), `--client-secrets`, `--keep-work`.

**Gemini bypass:** if `GEMINI_API_KEY` is present, `generate_text()` calls the official Google Gemini REST API directly instead of the local OpenCode service, so the OpenCode free-quota limit no longer applies. The key is read from `os.environ["GEMINI_API_KEY"]` first, then from the persistent Windows user/machine environment (so a key set after this terminal opened is still found), with whitespace and stray quotes trimmed. It is sent only in the `x-goog-api-key` header and never printed, logged, or written to a file. Default model: `gemini-3.8-flash` (`GEMINI_MODEL` overrides it). If a model is overloaded (429/500/502/503), missing (404), or stalls, the call fails over to `gemini-3.6-flash` → `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite`, and if every model is momentarily busy the whole ladder is retried once after a 20s pause. Each request is capped at 300s and one whole analysis call at 900s, so the API can never hang a run. Clip metadata records the model that actually answered. Without the key, behavior is unchanged.

**Low-spec safety (i5 5th gen / GT 730 class):** every yt-dlp and FFmpeg command runs with a hard timeout and stdin closed, so a stalled download or a slow encode is stopped instead of freezing the terminal — `--clips` defaults to a maximum of 5 (only genuine moments ship, 1..N), x264 uses `-preset veryfast -crf 20` with cheap `bicubic` scaling instead of lanczos, and yt-dlp gets `--socket-timeout 30`. Tune with `--threads N` (0 = auto; try `2` so the PC stays usable while rendering) and `--x264-preset ultrafast|superfast|veryfast|faster|fast|medium`. A typical run needs only the URL: `python viral_shorts.py "https://youtu.be/..."`.

Multi-clip runs (`--clips 3`..`5`) also extract a cover/thumbnail reference per clip: the analysis flags each moment's `peak` second (the biggest payoff) and FFmpeg grabs that frame as `<name>.jpg` next to the MP4, recorded in the JSON as `cover`. If one upload in a multi-clip run fails, the remaining clips still publish and the command exits non-zero with a per-clip failure list.

Unattended runs: passing the URL is all a scheduled or automated run needs — there are no prompts, and publishing to YouTube is the default (`--skip-upload` opts out):

```powershell
# e.g. Task Scheduler / cron / CI: analyze, render and publish with no interaction
python viral_shorts.py "https://www.youtube.com/watch?v=VIDEO_ID" --privacy public
```

The two spots that could otherwise stall an automated run fail fast with instructions instead: a run without a URL and without a terminal exits with an error rather than waiting for input, and a missing or expired `youtube_token.json` in a non-interactive run raises an error telling you to run the same command once from a terminal (the one-time browser consent), rather than opening a browser nobody is watching.

Notes: the AI must return valid JSON with exact `HH:MM:SS` timestamps inside the transcript, and the script forces the final cut into the 35-50 second window that opens on the hook. The six hashtags, the under-60-character `#Shorts` title, and the burnable ASCII hook are enforced in code even if the model drifts. Free-model usage follows the OpenCode free-tier quota; when it is exhausted the script stops with the reset time instead of failing blindly. Each YouTube upload costs 1600 API quota units (default 10000/day, about 6 uploads). The clipper never posts to Instagram, and you are responsible for the rights to the source video.

## Use with any AI assistant

Copy [GTA-CLIPPING-PROMPT.txt](GTA-CLIPPING-PROMPT.txt) into ChatGPT, Claude, Gemini, or another AI assistant. The prompt itself needs no API key, plugin, or special setup. If your AI can access your video and run local tools, it can help operate the clipper; otherwise it will tell you what it can do and guide you one step at a time. Automatic clipping still needs the free tool installed. YouTube chat scoring uses the same optional YouTube API key as live search; without it, audio and speech scoring still works.

For an AI that can access this project folder, [AI_WORKFLOW.md](AI_WORKFLOW.md) gives the detailed tool workflow. Never paste credentials into a prompt.

## Rights and platform notes

This release has no creator allowlist; the user chooses the source. You are responsible for obtaining rights and permissions to record, edit, and publish the footage. Creator credit does not replace permission. YouTube search ranking does not report a verified live concurrent-viewer leaderboard. This independent fan project is not affiliated with Rockstar Games, Take-Two Interactive, YouTube, Instagram, Meta, or Zernio.

## License

MIT. See [LICENSE](LICENSE). No OpenShorts application code or AutoCrop-Vertical code is bundled in this focused release.

## Several streams, several AI agents, several channels

See [docs/MULTI-AGENT.md](docs/MULTI-AGENT.md) for the roles (capture runner, checker, editor), running one capture per stream, the checker's veto rules, and the YouTube/Instagram daily limits. Posting to many channels with an equal share: [GTA Auto Sender](https://github.com/AnIntellectualBeing/GTA-Auto-Sender).
