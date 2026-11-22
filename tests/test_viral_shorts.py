import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viral_shorts import (
    CLIPS_DEFAULT,
    HASHTAGS,
    SECTION_TIMEOUT,
    SCALER_FLAGS,
    UPLOAD_GAP_DEFAULT,
    X264_CRF,
    X264_PRESET,
    ask_candidates,
    ask_with_fallback,
    build_parser,
    build_prompt,
    build_render_cmd,
    build_video_filter,
    chunk_cues,
    clips_overlap,
    cue_line,
    enforce_analysis,
    extract_json,
    fmt_ts,
    normalize_segment,
    parse_clip_list,
    parse_model_ref,
    parse_srt,
    parse_timestamp,
    provider_error_message,
    sanitize_filename,
    section_range,
    select_clips,
    trim_words,
    validate_analysis,
    write_ass,
    ytdlp_cmd,
    youtube_body,
)


class TimestampTests(unittest.TestCase):
    def test_fmt_ts_pads_hours(self):
        self.assertEqual(fmt_ts(0), "00:00:00")
        self.assertEqual(fmt_ts(92), "00:01:32")
        self.assertEqual(fmt_ts(3725), "01:02:05")

    def test_parse_timestamp_accepts_common_shapes(self):
        self.assertEqual(parse_timestamp(92), 92.0)
        self.assertEqual(parse_timestamp("92"), 92.0)
        self.assertEqual(parse_timestamp("15:20"), 920.0)
        self.assertEqual(parse_timestamp("00:15:20"), 920.0)
        self.assertEqual(parse_timestamp("00:15:20.500"), 920.5)
        self.assertEqual(parse_timestamp("00:00:01,000".replace(",", ".")), 1.0)

    def test_parse_timestamp_rejects_nonsense(self):
        for bad in ("abc", "1:2:3:4", "00:75:00", "", True):
            with self.assertRaises(ValueError):
                parse_timestamp(bad)


class SegmentTests(unittest.TestCase):
    def test_long_segment_is_trimmed_to_fifty_seconds(self):
        self.assertEqual(normalize_segment(100, 180, 600), (100, 150))

    def test_short_segment_is_extended_to_thirty_five_seconds(self):
        self.assertEqual(normalize_segment(100, 112, 600), (100, 135))

    def test_segment_near_the_end_shifts_back(self):
        self.assertEqual(normalize_segment(570, 615, 600), (555, 600))

    def test_video_shorter_than_target_uses_whole_video(self):
        self.assertEqual(normalize_segment(5, 20, 30), (0, 30))

    def test_equal_timestamps_get_a_minimum_window(self):
        start, end = normalize_segment(100, 100, 600)
        self.assertEqual(end - start, 35)

    def test_unknown_duration_keeps_length_in_range(self):
        self.assertEqual(normalize_segment(920, 965, None), (920, 965))


class TextHelperTests(unittest.TestCase):
    def test_trim_words_respects_limit(self):
        self.assertEqual(trim_words("a" * 95 + " extra", 90), "a" * 90)
        self.assertEqual(trim_words("  hello   world  ", 90), "hello world")

    def test_sanitize_filename_removes_windows_invalid_chars(self):
        self.assertEqual(sanitize_filename('GTA 6: "insane" <chase>?'), "GTA 6 insane chase")

    def test_section_range_matches_yt_dlp_syntax(self):
        self.assertEqual(section_range(920, 965), "*00:15:20-00:16:05")


class JsonExtractionTests(unittest.TestCase):
    def test_plain_json_object(self):
        self.assertEqual(extract_json('{"start": "00:01:00"}'), {"start": "00:01:00"})

    def test_fenced_json(self):
        text = "Here you go:\n```json\n{\"start\": \"00:01:00\", \"end\": \"00:01:45\"}\n```\nDone."
        self.assertEqual(extract_json(text)["end"], "00:01:45")

    def test_json_embedded_in_prose(self):
        text = 'Sure! {"title": "wow"} hope that helps'
        self.assertEqual(extract_json(text), {"title": "wow"})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("no object here")


class AnalysisContractTests(unittest.TestCase):
    def valid_raw(self):
        return {
            "start": "00:15:20",
            "end": "00:16:05",
            "title": "He DID NOT expect THIS in GTA 6",
            "description": "The chase went completely out of control.",
            "hashtags": ["#whatever"],
            "tags": ["gta6", "#Gaming", "gta6"],
            "score": "87",
            "reason": "screaming payoff",
        }

    def test_validate_requires_core_keys(self):
        with self.assertRaises(ValueError):
            validate_analysis({"start": "00:00:10"})

    def test_enforce_forces_exact_six_hashtags(self):
        result = enforce_analysis(self.valid_raw(), 3600)
        self.assertEqual(result["hashtags"], HASHTAGS)
        self.assertEqual(len(result["hashtags"]), 6)
        for tag in HASHTAGS:
            self.assertIn(tag, result["description"])

    def test_enforce_keeps_existing_hashtag_line_intact(self):
        raw = dict(self.valid_raw(), hashtags=HASHTAGS)
        raw["description"] = "Epic.\n\n" + " ".join(HASHTAGS)
        result = enforce_analysis(raw, 3600)
        self.assertEqual(result["description"].count("#Shorts"), 1)

    def test_enforce_segment_length_and_title_limit(self):
        result = enforce_analysis(self.valid_raw(), 3600)
        self.assertEqual(result["end"] - result["start"], 45)
        self.assertLessEqual(len(result["title"]), 60)
        self.assertIn("#Shorts", result["title"])

    def test_enforce_title_survives_trim_and_still_carries_shorts_tag(self):
        raw = dict(self.valid_raw(), title="Wait for the GTA 6 physics fail " + "x" * 40 + " #Shorts")
        result = enforce_analysis(raw, 3600)
        self.assertLessEqual(len(result["title"]), 60)
        self.assertIn("#Shorts", result["title"])

    def test_enforce_hook_is_ascii_and_burnable(self):
        raw = dict(self.valid_raw(), hook='WAIT "FOR" IT... \U0001F480' + "x" * 40)
        result = enforce_analysis(raw, 3600)
        hook = result["hook"]
        self.assertTrue(hook)
        self.assertLessEqual(len(hook), 40)
        self.assertTrue(all(" " <= ch <= "~" for ch in hook))  # ASCII only, burns cleanly
        self.assertNotIn('"', hook)

    def test_enforce_hook_defaults_to_empty_when_missing(self):
        raw = dict(self.valid_raw())
        raw.pop("hook", None)
        self.assertEqual(enforce_analysis(raw, 3600)["hook"], "")

    def test_enforce_cleans_and_budgets_tags(self):
        result = enforce_analysis(self.valid_raw(), 3600)
        self.assertEqual(result["tags"], ["gta6", "Gaming"])  # dedupe, '#' stripped
        huge = dict(self.raw_with_tags(["x" * 100] * 10), score=1)
        result = enforce_analysis(huge, 3600)
        self.assertLessEqual(sum(len(t) + 1 for t in result["tags"]), 480)

    def raw_with_tags(self, tags):
        raw = self.valid_raw()
        raw["tags"] = tags
        return raw

    def test_enforce_falls_back_when_tags_missing(self):
        raw = dict(self.valid_raw())
        raw["tags"] = []
        result = enforce_analysis(raw, 3600)
        self.assertTrue(result["tags"])

    def test_score_is_a_float(self):
        self.assertEqual(enforce_analysis(self.valid_raw(), 3600)["score"], 87.0)

    def test_peak_is_clamped_into_the_window(self):
        self.assertEqual(enforce_analysis(dict(self.valid_raw(), peak="00:00:05"), 3600)["peak"], 920)
        self.assertEqual(enforce_analysis(dict(self.valid_raw(), peak="00:59:00"), 3600)["peak"], 965)

    def test_peak_defaults_to_the_window_midpoint(self):
        peak = enforce_analysis(self.valid_raw(), 3600)["peak"]
        self.assertTrue(920 <= peak <= 965)


class ProviderErrorTests(unittest.TestCase):
    def test_rate_limit_message_mentions_reset_and_go(self):
        import json as _json
        from datetime import datetime, timedelta
        body = _json.dumps({"reason": "rate_limit", "message": "Free usage limit reached.",
                            "status": 429, "retryAfterMs": 3600000})
        message = provider_error_message(502, body)
        self.assertIn("free usage limit", message.lower())
        self.assertIn("opencode.ai/go", message)
        expected = (datetime.now().astimezone() + timedelta(milliseconds=3600000)).strftime("%Y-%m-%d")
        self.assertIn(expected, message)

    def test_generic_message_is_passed_through(self):
        body = '{"message": "upstream exploded", "status": 502}'
        self.assertIn("upstream exploded", provider_error_message(502, body))

    def test_unparseable_body_falls_back_to_status(self):
        self.assertIn("502", provider_error_message(502, "<html>bad gateway</html>"))


class SrtTests(unittest.TestCase):
    def test_parses_multiline_cues_with_crlf(self):
        content = (
            "1\r\n00:00:01,000 --> 00:00:04,000\r\nline one\r\nline two\r\n\r\n"
            "2\r\n00:00:05,500 --> 00:00:08,250\r\nsecond cue\r\n\r\n"
        )
        cues = parse_srt(content)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["start"], 1.0)
        self.assertEqual(cues[0]["text"], "line one line two")
        self.assertEqual(cues[1]["end"], 8.25)

    def test_sorts_out_of_order_cues(self):
        content = (
            "1\n00:01:00,000 --> 00:01:02,000\nlate\n\n"
            "2\n00:00:01,000 --> 00:00:02,000\nearly\n\n"
        )
        cues = parse_srt(content)
        self.assertEqual([c["text"] for c in cues], ["early", "late"])

    def test_skips_blocks_without_valid_times(self):
        cues = parse_srt("garbage\nnot a timestamp\n")
        self.assertEqual(cues, [])


class ChunkingTests(unittest.TestCase):
    def make_cues(self, count):
        return [
            {"start": float(i * 3), "end": float(i * 3 + 2.5), "text": f"cue {i} " + "word " * 8}
            for i in range(count)
        ]

    def test_short_transcript_is_one_window(self):
        cues = self.make_cues(5)
        self.assertEqual(len(chunk_cues(cues, max_chars=100000)), 1)

    def test_long_transcript_splits_with_overlap(self):
        cues = self.make_cues(400)
        windows = chunk_cues(cues, max_chars=3000)
        self.assertGreater(len(windows), 1)
        seen = {id(cue) for window in windows for cue in window}
        self.assertEqual(seen, {id(cue) for cue in cues})  # every cue is analyzed somewhere
        self.assertLess(windows[1][0]["start"], windows[0][-1]["start"])  # windows overlap in time

    def test_cue_line_has_absolute_timestamp(self):
        self.assertEqual(cue_line({"start": 920.0, "end": 922.0, "text": "  hi  there "}),
                         "[00:15:20] hi there")


class PromptTests(unittest.TestCase):
    def test_prompt_contains_contract(self):
        prompt = build_prompt("[00:00:10] hello", 1, 1, "Some Video", 600)
        self.assertIn('"start": "HH:MM:SS"', prompt)
        self.assertIn("35 and 50 seconds", prompt)
        self.assertIn(" ".join(HASHTAGS), prompt)
        self.assertIn("[00:00:10] hello", prompt)
        self.assertIn("00:10:00", prompt)  # duration hint

    def test_prompt_carries_the_pro_editor_persona_and_hook_field(self):
        prompt = build_prompt("x", 1, 1, None, None)
        # the exact high-retention editor brief this engine was specced with
        self.assertIn(
            "Analyze this gaming stream transcript like a pro viral short-form editor.",
            prompt,
        )
        self.assertIn("starts immediately with a strong visual or verbal hook", prompt)
        self.assertIn('"hook"', prompt)
        self.assertIn("ending with #Shorts", prompt)
        self.assertIn("60 characters", prompt)

    def test_single_prompt_stays_a_single_object(self):
        prompt = build_prompt("x", 1, 1, None, None)
        self.assertIn('"peak"', prompt)
        self.assertNotIn('"clips"', prompt)

    def test_multi_prompt_requests_a_ranked_distinct_clip_list(self):
        prompt = build_prompt("x", 1, 1, None, None, count=5)
        self.assertIn('"clips"', prompt)
        self.assertIn("ONE clip object per distinct strong moment, up to 5 total", prompt)
        self.assertIn("never pad the list", prompt)
        self.assertIn("score it 65 or higher", prompt)
        self.assertIn("Scan the ENTIRE", prompt)
        self.assertIn("DISTINCT", prompt)
        self.assertIn("beginning, middle, and end", prompt)
        self.assertIn("title must be unique", prompt)
        self.assertIn('"peak"', prompt)
        self.assertIn("35 and 50 seconds", prompt)

    def test_multi_prompt_never_asks_for_a_single_moment(self):
        prompt = build_prompt("x", 1, 1, None, None, count=5)
        self.assertNotIn("SINGLE highest-retention", prompt)
        self.assertIn("one clip per distinct strong scene", prompt)

    def test_single_prompt_still_asks_for_one_moment(self):
        prompt = build_prompt("x", 1, 1, None, None)
        self.assertIn("SINGLE highest-retention", prompt)
        self.assertNotIn("one clip per distinct strong scene", prompt)

    def test_multi_window_prompt_marks_the_part(self):
        prompt = build_prompt("x", 2, 3, None, None)
        self.assertIn("part 2 of 3", prompt)


class UploadPayloadTests(unittest.TestCase):
    def meta(self):
        return {
            "title": "T" * 120,
            "description": "desc",
            "tags": ["a", "b"],
        }

    def test_body_limits_and_privacy(self):
        body = youtube_body(self.meta(), "unlisted")
        self.assertLessEqual(len(body["snippet"]["title"]), 60)
        self.assertIn("#Shorts", body["snippet"]["title"])
        self.assertEqual(body["status"]["privacyStatus"], "unlisted")
        self.assertEqual(body["status"]["selfDeclaredMadeForKids"], False)
        self.assertEqual(body["snippet"]["categoryId"], "20")
        self.assertEqual(body["snippet"]["tags"], ["a", "b"])


class ModelRefTests(unittest.TestCase):
    def test_parses_provider_and_id(self):
        self.assertEqual(parse_model_ref("opencode/mimo-v2.6-flash-free"),
                         {"providerID": "opencode", "id": "mimo-v2.6-flash-free"})

    def test_parses_variant(self):
        self.assertEqual(parse_model_ref("openai/gpt-x#fast"),
                         {"providerID": "openai", "id": "gpt-x", "variant": "fast"})

    def test_rejects_missing_model(self):
        from viral_shorts import Fail
        with self.assertRaises(Fail):
            parse_model_ref("opencode")


class AssWritingTests(unittest.TestCase):
    def test_writes_relative_times_and_strips_braces(self):
        cues = [
            {"start": 921.0, "end": 923.0, "text": "normal text"},
            {"start": 919.0, "end": 922.0, "text": "{\\bad} text"},
            {"start": 970.0, "end": 975.0, "text": "outside window"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.ass"
            write_ass(path, cues, 920, 965)
            content = path.read_text(encoding="utf-8")
        self.assertIn("PlayResX: 1080", content)
        self.assertIn("PlayResY: 1920", content)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:03.00", content)   # 921-920 .. 923-920
        self.assertIn("normal text", content)
        self.assertIn("(bad) text", content)                          # braces stripped
        self.assertNotIn("outside window", content)                   # outside the segment

    def test_hook_overlay_is_burned_top_center_for_the_whole_clip(self):
        cues = [{"start": 921.0, "end": 923.0, "text": "hi"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.ass"
            write_ass(path, cues, 920, 965, hook="WAIT FOR IT")
            content = path.read_text(encoding="utf-8")
        self.assertIn("Style: Hook,", content)                        # dedicated top style
        hook_line = next(line for line in content.splitlines() if ",Hook," in line)
        self.assertTrue(hook_line.startswith("Dialogue: 0,0:00:00.00,0:00:45.00"))
        self.assertIn("WAIT FOR IT", hook_line)

    def test_no_hook_line_when_the_ai_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.ass"
            write_ass(path, [], 920, 965, hook="")
            content = path.read_text(encoding="utf-8")
        self.assertNotIn(",Hook,", content)


class UrlResolutionTests(unittest.TestCase):
    def test_direct_video_links_pass_through_untouched(self):
        from viral_shorts import resolve_video_url

        for url in (
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "https://www.youtube.com/shorts/abc123",
            "https://www.youtube.com/live/abc123",
        ):
            self.assertEqual(resolve_video_url(url), url)

    def test_channel_link_resolves_to_newest_upload(self):
        import viral_shorts
        from viral_shorts import resolve_video_url

        fake = mock.Mock(returncode=0, stdout=json.dumps({"entries": [{"id": "NEW123"}]}))
        with mock.patch.object(viral_shorts, "run", return_value=fake), \
             mock.patch("sys.stdout", new=io.StringIO()):
            resolved = resolve_video_url("https://www.youtube.com/@somechannel")
        self.assertEqual(resolved, "https://www.youtube.com/watch?v=NEW123")

    def test_unresolvable_link_fails_with_a_clear_message(self):
        import viral_shorts
        from viral_shorts import Fail, resolve_video_url

        fake = mock.Mock(returncode=1, stdout="", stderr="unsupported url")
        with mock.patch.object(viral_shorts, "run", return_value=fake):
            with self.assertRaises(Fail) as ctx:
                resolve_video_url("https://example.com/not-youtube")
        self.assertIn("not a video", str(ctx.exception))


class UnattendedRunTests(unittest.TestCase):
    """Automated runs must never hang on a prompt or a browser nobody can see."""

    def test_main_exits_cleanly_without_url_when_non_interactive(self):
        import viral_shorts

        with mock.patch.object(sys, "stdin", io.StringIO("")), \
             mock.patch.object(viral_shorts, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(viral_shorts, "check_ass_filter"), \
             mock.patch.object(viral_shorts, "opencode_service", return_value=("http://127.0.0.1:1", "")), \
             mock.patch.object(sys, "argv", ["viral_shorts.py"]):
            with self.assertRaises(SystemExit) as ctx:
                viral_shorts.main()
        self.assertIn("non-interactively", str(ctx.exception))

    def test_upload_short_fails_fast_instead_of_opening_a_browser(self):
        import viral_shorts
        from viral_shorts import Fail

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            secrets = tmp / "client_secrets.json"
            secrets.write_text("{}", encoding="utf-8")
            with mock.patch.object(viral_shorts, "YOUTUBE_TOKEN", tmp / "absent.json"), \
                 mock.patch.object(sys, "stdin", io.StringIO("")):
                with self.assertRaises(Fail) as ctx:
                    viral_shorts.upload_short(tmp / "clip.mp4", {}, "unlisted", str(secrets))
        self.assertIn("non-interactive", str(ctx.exception))


class MultiClipTests(unittest.TestCase):
    @staticmethod
    def clip(start, end, score=80):
        return {"start": start, "end": end, "score": score}

    # --- select_clips: dynamic 1..N, never padded ---
    def test_weak_candidates_are_never_shipped(self):
        cands = [self.clip(0, 45, 90), self.clip(100, 145, 88),
                 self.clip(200, 245, 86), self.clip(300, 345, 40)]
        picked = select_clips(cands, count=5, floor=65)
        self.assertEqual(len(picked), 3)                      # 40-score clip dropped
        self.assertTrue(all(c["score"] >= 65 for c in picked))

    def test_two_strong_moments_ship_two_clips_not_five(self):
        cands = [self.clip(0, 45, 90), self.clip(100, 145, 88),
                 self.clip(200, 245, 40), self.clip(300, 345, 30)]
        picked = select_clips(cands, count=5, floor=65)
        self.assertEqual([c["score"] for c in picked], [90, 88])   # no backfill

    def test_one_strong_moment_ships_exactly_one_clip(self):
        cands = [self.clip(0, 45, 93), self.clip(100, 145, 50), self.clip(200, 245, 20)]
        picked = select_clips(cands, count=5, floor=65)
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["score"], 93)

    def test_when_nothing_clears_the_floor_the_best_clip_still_ships(self):
        cands = [self.clip(0, 45, 44), self.clip(100, 145, 61), self.clip(200, 245, 30)]
        picked = select_clips(cands, count=5, floor=65)
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["score"], 61)              # best available, never empty

    def test_duplicate_titles_are_collapsed_to_one_clip(self):
        first = dict(self.clip(0, 45, 92), title="Same Title #Shorts")
        second = dict(self.clip(200, 245, 90), title="same title #shorts")
        picked = select_clips([first, second], count=5, floor=65)
        self.assertEqual(len(picked), 1)

    def test_overlapping_and_too_close_candidates_are_dropped(self):
        cands = [self.clip(0, 45, 90), self.clip(48, 93, 99),   # only 3s apart
                 self.clip(100, 145, 86), self.clip(200, 245, 84)]
        picked = select_clips(cands, count=5, floor=65)
        # best-first: the 99-score clip wins, its 3s-away neighbour is dropped
        self.assertEqual([c["start"] for c in picked], [48, 100, 200])
        for i, a in enumerate(picked):
            for b in picked[i + 1:]:
                self.assertFalse(clips_overlap(a, b))         # zero overlap

    def test_results_are_ranked_best_first(self):
        cands = [self.clip(0, 45, 70), self.clip(100, 145, 95), self.clip(200, 245, 80)]
        picked = select_clips(cands, count=3, floor=65)
        self.assertEqual([c["score"] for c in picked], [95, 80, 70])

    def test_selection_spreads_across_the_timeline_when_slots_are_limited(self):
        # three near-perfect clips clustered at the start, plus two solid ones later
        cands = [self.clip(100, 145, 99), self.clip(150, 195, 98), self.clip(200, 245, 97),
                 self.clip(600, 645, 80), self.clip(900, 945, 70)]
        picked = select_clips(cands, count=3, floor=65, duration=1000)
        self.assertEqual([c["start"] for c in picked], [100, 600, 900])  # start/middle/end
        for i, a in enumerate(picked):
            for b in picked[i + 1:]:
                self.assertFalse(clips_overlap(a, b))

    def test_timeline_spread_is_still_ranked_best_first(self):
        cands = [self.clip(100, 145, 70), self.clip(600, 645, 95), self.clip(900, 945, 85)]
        picked = select_clips(cands, count=3, floor=65, duration=1000)
        self.assertEqual([c["score"] for c in picked], [95, 85, 70])
        self.assertEqual({c["start"] for c in picked}, {100, 600, 900})   # still spread

    def test_without_a_duration_selection_is_purely_best_first(self):
        cands = [self.clip(100, 145, 99), self.clip(150, 195, 98), self.clip(200, 245, 97),
                 self.clip(600, 645, 80), self.clip(900, 945, 70)]
        picked = select_clips(cands, count=3, floor=65)
        self.assertEqual([c["start"] for c in picked], [100, 150, 200])

    def test_clips_overlap_requires_a_five_second_gap(self):
        self.assertFalse(clips_overlap(self.clip(0, 45), self.clip(50, 95)))  # exactly 5s
        self.assertTrue(clips_overlap(self.clip(0, 45), self.clip(49, 94)))   # under 5s
        self.assertTrue(clips_overlap(self.clip(40, 85), self.clip(60, 105)))  # overlapping

    # --- parse_clip_list ---
    def test_parse_clip_list_accepts_every_reply_shape(self):
        one = {"start": "00:00:10", "end": "00:00:50", "title": "t"}
        two = {"start": "00:02:00", "end": "00:02:45", "title": "u"}
        shapes = [
            json.dumps({"clips": [one, two]}),
            json.dumps([one, two]),
            json.dumps(one),
            "Sure!\n```json\n" + json.dumps({"clips": [one]}) + "\n```",
        ]
        for reply in shapes:
            items = parse_clip_list(reply)
            self.assertTrue(any(isinstance(i, dict) and "start" in i for i in items), reply)

    def test_parse_clip_list_rejects_garbage(self):
        for reply in ("no json here", '{"foo": 1}', '["a", "b"]'):
            with self.assertRaises(ValueError):
                parse_clip_list(reply)

    # --- ask_candidates ---
    def test_ask_candidates_caps_at_the_requested_count(self):
        items = {"clips": [
            {"start": f"00:0{i}:00", "end": f"00:0{i}:45", "title": "t", "description": "d"}
            for i in range(5)
        ]}
        with mock.patch("viral_shorts.generate_text", return_value=json.dumps(items)):
            got = ask_candidates("p", parse_model_ref("opencode/mimo-v2.6-flash-free"), 3)
        self.assertEqual(len(got), 3)

    def test_ask_candidates_retries_then_fails_on_bad_json(self):
        from viral_shorts import Fail
        with mock.patch("viral_shorts.generate_text",
                        side_effect=["garbage", "more garbage", "still garbage"]):
            with self.assertRaises(Fail):
                ask_candidates("p", parse_model_ref("opencode/mimo-v2.6-flash-free"), 3)

    # --- ask_with_fallback ---
    def test_fallback_switches_on_quota_error(self):
        from viral_shorts import Fail, ask_with_fallback, fmt_model
        primary = parse_model_ref("opencode/mimo-v2.6-flash-free")
        seen = []

        def call(prompt, member):
            seen.append(fmt_model(member))
            if fmt_model(member) == "opencode/mimo-v2.6-flash-free":
                raise Fail("OpenCode's free usage limit for free models is reached.")
            return "ok"

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
             mock.patch("viral_shorts._windows_persistent_env", return_value=""):
            result, member = ask_with_fallback(
                "p", primary, ["opencode/nemotron-3.5-super-vl-free"], call,
            )
        self.assertEqual(result, "ok")
        self.assertEqual(fmt_model(member), "opencode/nemotron-3.5-super-vl-free")
        self.assertEqual(len(seen), 2)                          # mimo, then nemotron

    def test_fallback_is_not_used_for_unrelated_errors(self):
        from viral_shorts import Fail, ask_with_fallback
        attempts = []

        def call(prompt, member):
            attempts.append(member)
            raise Fail("model opencode/nope was not found")

        with self.assertRaises(Fail) as ctx:
            ask_with_fallback(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"),
                ["opencode/deepseek-v4.1-flash-free"], call,
            )
        self.assertIn("not found", str(ctx.exception))
        self.assertEqual(len(attempts), 1)

    def test_fallback_raises_the_quota_error_when_every_model_is_limited(self):
        from viral_shorts import Fail, ask_with_fallback

        def call(prompt, member):
            raise Fail("free usage limit reached")

        with self.assertRaises(Fail):
            ask_with_fallback(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"),
                ["opencode/nemotron-3.5-super-vl-free", "opencode/deepseek-v4.1-flash-free"],
                call,
            )


class GeminiBypassTests(unittest.TestCase):
    """GEMINI_API_KEY set -> generate_text must bypass the OpenCode service entirely."""

    KEY = "unit-test-key-abc123"

    @staticmethod
    def ok(text):
        resp = mock.Mock(status_code=200, text="{}")
        resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return resp

    @staticmethod
    def error(status, body='{"error": {"message": "boom"}}'):
        resp = mock.Mock(status_code=status, text=body)
        resp.json.side_effect = ValueError("not json")
        return resp

    def test_key_set_calls_gemini_and_keeps_key_out_of_the_url(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("  hello gemini  ")) as post:
            got = viral_shorts.generate_text(
                "prompt", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(got, "hello gemini")                 # stripped, same return format
        args, kwargs = post.call_args
        self.assertIn("generativelanguage.googleapis.com", args[0])
        self.assertNotIn(self.KEY, args[0])                   # key never in the URL
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], self.KEY)

    def test_no_key_keeps_the_opencode_path_untouched(self):
        import viral_shorts

        resp = mock.Mock(status_code=200, text="{}")
        resp.json.return_value = {"data": {"text": " opencode says hi "}}
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
             mock.patch.object(viral_shorts, "_windows_persistent_env", return_value=""), \
             mock.patch.object(viral_shorts, "opencode_service",
                               return_value=("http://127.0.0.1:1", "pw")), \
             mock.patch.object(viral_shorts.requests, "post", return_value=resp) as post:
            got = viral_shorts.generate_text(
                "prompt", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(got, "opencode says hi")
        self.assertIn("/api/experimental/generate/text", post.call_args[0][0])

    def test_unknown_model_falls_through_the_gemini_ladder(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=[self.error(404), self.ok("second model works")]) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(got, "second model works")
        urls = [call.args[0] for call in post.call_args_list]
        self.assertEqual(len(urls), 2)
        self.assertIn(viral_shorts.GEMINI_MODEL_LADDER[0], urls[0])
        self.assertIn(viral_shorts.GEMINI_MODEL_LADDER[1], urls[1])

    def test_overloaded_model_rolls_to_the_next_gemini_id(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.time, "sleep"), \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=[self.error(503), self.ok("next model answered")]) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=1)
        self.assertEqual(got, "next model answered")
        urls = [call.args[0] for call in post.call_args_list]
        self.assertEqual(len(urls), 2)
        self.assertIn(viral_shorts.GEMINI_MODEL_LADDER[1], urls[1])

    def test_ladder_has_no_dead_model_ids(self):
        import viral_shorts

        self.assertNotIn("gemini-2.5-flash", viral_shorts.GEMINI_MODEL_LADDER)
        self.assertEqual(viral_shorts.GEMINI_MODEL_LADDER[0], "gemini-3.8-flash")
        self.assertEqual(len(set(viral_shorts.GEMINI_MODEL_LADDER)),
                         len(viral_shorts.GEMINI_MODEL_LADDER))

    def test_a_fleet_wide_spike_is_retried_once(self):
        import viral_shorts

        ladder_size = len(viral_shorts.GEMINI_MODEL_LADDER)
        busy = [self.error(503)] * ladder_size
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.time, "sleep") as sleep, \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=busy + [self.ok("second pass worked")]) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=1)
        self.assertEqual(got, "second pass worked")
        self.assertEqual(len(post.call_args_list), ladder_size + 1)
        self.assertTrue(sleep.called)                 # paused before the second pass

    def test_a_permanent_outage_still_fails_within_the_budget(self):
        import viral_shorts
        from viral_shorts import Fail

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.time, "sleep"), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.error(503)):
            with self.assertRaises(Fail):
                viral_shorts.generate_text(
                    "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=1)

    def test_env_model_override_wins(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY,
                                          "GEMINI_MODEL": "gemini-custom-x"}), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("ok")) as post:
            viral_shorts.generate_text("p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertIn("gemini-custom-x", post.call_args[0][0])

    def test_error_message_never_leaks_the_key(self):
        import viral_shorts
        from viral_shorts import Fail

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.error(500, "internal server error")), \
             mock.patch.object(viral_shorts.time, "sleep"):
            with self.assertRaises(Fail) as ctx:
                viral_shorts.generate_text(
                    "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=2)
        self.assertNotIn(self.KEY, str(ctx.exception))
        self.assertIn("500", str(ctx.exception))

    def test_rejected_key_fails_with_a_clear_message(self):
        import viral_shorts
        from viral_shorts import Fail

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts.requests, "post", return_value=self.error(403)):
            with self.assertRaises(Fail) as ctx:
                viral_shorts.generate_text(
                    "p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertNotIn(self.KEY, str(ctx.exception))

    def test_text_part_handles_missing_candidates(self):
        from viral_shorts import gemini_text_part

        self.assertEqual(gemini_text_part({}), "")
        self.assertEqual(gemini_text_part({"candidates": []}), "")
        self.assertEqual(
            gemini_text_part({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}),
            "ab",
        )


class GeminiKeyRobustnessTests(unittest.TestCase):
    """The bypass check must never miss a key, and must never consult OpenCode."""

    KEY = "unit-test-key-abc123"

    @staticmethod
    def ok(text):
        resp = mock.Mock(status_code=200, text="{}")
        resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
        return resp

    @staticmethod
    def error(status):
        resp = mock.Mock(status_code=status, text='{"error": {"message": "x"}}')
        resp.json.side_effect = ValueError("not json")
        return resp

    def test_key_is_trimmed_and_unquoted(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": '  "  abc123  "  '}):
            self.assertEqual(viral_shorts.gemini_key(), "abc123")
        self.assertEqual(viral_shorts.clean_key(None), "")
        self.assertEqual(viral_shorts.clean_key(" 'k' \n"), "k")

    def test_persistent_windows_env_is_used_when_process_env_is_empty(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
             mock.patch.object(viral_shorts, "_windows_persistent_env",
                               return_value="registry-key"):
            self.assertEqual(viral_shorts.gemini_key(), "registry-key")
            self.assertEqual(viral_shorts.gemini_key_source(),
                             "Windows user/machine environment")

    def test_generate_text_uses_the_registry_key_when_process_env_is_empty(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
             mock.patch.object(viral_shorts, "_windows_persistent_env",
                               return_value="registry-key"), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("from registry")) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(got, "from registry")
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "registry-key")

    def test_registry_reader_queries_user_and_machine_hives(self):
        import viral_shorts

        fake = mock.MagicMock()
        fake.HKEY_CURRENT_USER = "HKCU"
        fake.HKEY_LOCAL_MACHINE = "HKLM"
        fake.OpenKey.return_value.__enter__.return_value = "handle"
        fake.QueryValueEx.return_value = ("registry-key", 1)
        with mock.patch.dict(sys.modules, {"winreg": fake}), \
             mock.patch.object(viral_shorts.os, "name", "nt"):
            self.assertEqual(
                viral_shorts._windows_persistent_env("GEMINI_API_KEY"), "registry-key")
        self.assertTrue(fake.QueryValueEx.called)

    def test_key_present_never_touches_the_opencode_service(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts, "opencode_service",
                               side_effect=AssertionError("OpenCode must not be consulted")), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("gemini answered")):
            self.assertEqual(
                viral_shorts.generate_text(
                    "p", parse_model_ref("opencode/mimo-v2.6-flash-free")),
                "gemini answered")

    def test_fallback_chain_is_disabled_when_a_key_exists(self):
        import viral_shorts
        from viral_shorts import Fail, ask_with_fallback

        calls = []

        def call(prompt, member):
            calls.append(member)
            raise Fail("usage limit hit")   # would normally trigger an OpenCode fallback

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}):
            with self.assertRaises(Fail):
                ask_with_fallback(
                    "p", parse_model_ref("opencode/mimo-v2.6-flash-free"),
                    ["opencode/nemotron-3.5-super-vl-free"], call,
                )
        self.assertEqual(len(calls), 1)     # no OpenCode model was tried

    def test_bad_model_override_still_lands_on_the_gemini_default(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY,
                                          "GEMINI_MODEL": "gemini-does-not-exist"}), \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=[self.error(404), self.ok("recovered")]) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(got, "recovered")
        urls = [call.args[0] for call in post.call_args_list]
        self.assertIn("gemini-does-not-exist", urls[0])
        self.assertIn(viral_shorts.GEMINI_MODEL_LADDER[0], urls[1])

    def test_default_model_is_gemini_3_8_flash(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("ok")) as post:
            viral_shorts.generate_text("p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertIn("gemini-3.8-flash", post.call_args.args[0])

    def test_metadata_names_the_model_that_actually_answered(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=[self.error(503), self.ok("x")]):
            viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"))
        self.assertEqual(
            viral_shorts.active_model_label(
                parse_model_ref("opencode/mimo-v2.6-flash-free")),
            viral_shorts.GEMINI_MODEL_LADDER[1],
        )

    def test_a_stalled_model_rolls_to_the_next_gemini_id(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY, "GEMINI_MODEL": ""}), \
             mock.patch.object(viral_shorts.requests, "post",
                               side_effect=[viral_shorts.requests.Timeout(),
                                            self.ok("after timeout")]) as post:
            got = viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=1)
        self.assertEqual(got, "after timeout")
        self.assertEqual(len(post.call_args_list), 2)

    def test_each_request_timeout_is_bounded(self):
        import viral_shorts

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts.requests, "post",
                               return_value=self.ok("ok")) as post:
            viral_shorts.generate_text(
                "p", parse_model_ref("opencode/mimo-v2.6-flash-free"), attempts=1)
        connect, read = post.call_args.kwargs["timeout"]
        self.assertLessEqual(connect, viral_shorts.GEMINI_TIMEOUT[0])
        self.assertLessEqual(read, viral_shorts.GEMINI_TIMEOUT[1])

    def test_main_preflight_skips_opencode_when_a_key_exists(self):
        import viral_shorts

        buffer = io.StringIO()
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": self.KEY}), \
             mock.patch.object(viral_shorts, "find_ffmpeg", return_value="ffmpeg"), \
             mock.patch.object(viral_shorts, "check_ass_filter"), \
             mock.patch.object(viral_shorts, "opencode_service",
                               side_effect=AssertionError("must not be called")), \
             mock.patch.object(sys, "stdin", io.StringIO("")), \
             mock.patch.object(sys, "argv", ["viral_shorts.py"]), \
             mock.patch.object(sys, "stdout", buffer):
            with self.assertRaises(SystemExit):
                viral_shorts.main()
        self.assertIn("Gemini", buffer.getvalue())
        self.assertIn("OpenCode quota is not used", buffer.getvalue())


class LowSpecTests(unittest.TestCase):
    """i5 5th gen / GT 730 class: bounded commands, cheap encode, URL-only CLI."""

    # --- run(): nothing may hang ---
    def test_run_passes_stdin_devnull_and_the_timeout_through(self):
        import viral_shorts

        fake = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(viral_shorts.subprocess, "run", return_value=fake) as proc:
            viral_shorts.run(["ffmpeg", "-i", "x"], timeout=42, cwd=".")
        args, kwargs = proc.call_args
        self.assertEqual(args[0], ["ffmpeg", "-i", "x"])
        self.assertEqual(kwargs["timeout"], 42)
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["capture_output"])

    def test_a_stalled_command_becomes_a_clear_failure_not_a_hang(self):
        import viral_shorts
        from viral_shorts import Fail

        expired = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)
        with mock.patch.object(viral_shorts.subprocess, "run", side_effect=expired):
            with self.assertRaises(Fail) as ctx:
                viral_shorts.run(["ffmpeg", "-y", "x"], timeout=10)
        self.assertIn("ffmpeg", str(ctx.exception))
        self.assertIn("stopped", str(ctx.exception))

    def test_section_download_is_time_boxed(self):
        import viral_shorts
        from viral_shorts import Fail

        fake = mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(viral_shorts, "run", return_value=fake) as proc:
                with self.assertRaises(Fail):
                    viral_shorts.download_section(
                        "https://youtu.be/x", 100, 145, tmp, {"is_live": False})
        self.assertGreaterEqual(len(proc.call_args_list), 1)
        for call in proc.call_args_list:
            self.assertEqual(call.kwargs.get("timeout"), SECTION_TIMEOUT)

    def test_every_ytdlp_call_has_a_socket_timeout(self):
        cmd = ytdlp_cmd("--dump-single-json", "u")
        idx = cmd.index("--socket-timeout")
        self.assertEqual(cmd[idx + 1], "30")

    # --- render command: cheap and non-blocking ---
    def test_render_cmd_is_lightweight_and_cannot_wait_on_input(self):
        cmd = build_render_cmd("ffmpeg", "section.mp4", "final.mp4",
                               "[0:v]x[v]", ["-map", "[v]"], 45, True)
        self.assertIn("-nostdin", cmd)
        self.assertEqual(cmd[cmd.index("-preset") + 1], X264_PRESET)
        self.assertEqual(cmd[cmd.index("-crf") + 1], str(X264_CRF))
        self.assertEqual(cmd[cmd.index("-threads") + 1], "0")   # auto
        self.assertIn("+faststart", cmd)                        # YouTube-friendly MP4

    def test_render_cmd_honours_speed_overrides(self):
        cmd = build_render_cmd("ffmpeg", "s.mp4", "o.mp4", "[0:v]x[v]",
                               ["-map", "[v]"], 45, False,
                               threads=2, preset="ultrafast", crf=23)
        self.assertEqual(cmd[cmd.index("-preset") + 1], "ultrafast")
        self.assertEqual(cmd[cmd.index("-threads") + 1], "2")
        self.assertEqual(cmd[cmd.index("-crf") + 1], "23")
        self.assertIn("-an", cmd)

    def test_scale_filter_uses_the_cheap_scaler(self):
        chain = build_video_filter("captions.ass")
        self.assertIn(f"flags={SCALER_FLAGS}", chain)
        self.assertIn("scale=1080:1920", chain)
        self.assertIn("ass=captions.ass", chain)
        self.assertIn("crop=", chain)

    # --- CLI: one argument is a complete run ---
    def test_cover_falls_back_to_another_frame_instead_of_failing(self):
        import viral_shorts

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            target = Path(cmd[-1])
            if len(calls) == 1:
                return mock.Mock(returncode=1, stderr="no frame")
            target.write_bytes(b"\xff\xd8" + b"0" * 4096)   # a plausible JPEG
            return mock.Mock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cover.jpg"
            with mock.patch.object(viral_shorts, "run", side_effect=fake_run):
                got = viral_shorts.extract_cover(
                    "ffmpeg", "final.mp4", 611, 575, out, length=45)
        self.assertEqual(got, out)
        self.assertEqual(len(calls), 2)      # peak frame failed, midpoint worked

    def test_cover_returns_none_when_no_frame_can_be_grabbed(self):
        import viral_shorts

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cover.jpg"
            with mock.patch.object(viral_shorts, "run",
                                   return_value=mock.Mock(returncode=1, stderr="boom")):
                self.assertIsNone(viral_shorts.extract_cover(
                    "ffmpeg", "final.mp4", 611, 575, out, length=45))

    def test_bare_url_defaults_to_three_to_five_public_clips(self):
        args = build_parser().parse_args(["https://youtu.be/x"])
        self.assertEqual(args.clips, CLIPS_DEFAULT)
        self.assertEqual(args.privacy, "public")
        self.assertEqual(args.upload_gap, UPLOAD_GAP_DEFAULT)
        self.assertEqual(args.threads, 0)
        self.assertEqual(args.x264_preset, X264_PRESET)
        self.assertFalse(args.skip_upload)
        self.assertFalse(args.dry_run)

    def test_low_spec_overrides_parse(self):
        args = build_parser().parse_args(
            ["u", "--clips", "1", "--threads", "2", "--x264-preset", "ultrafast"])
        self.assertEqual(args.clips, 1)
        self.assertEqual(args.threads, 2)
        self.assertEqual(args.x264_preset, "ultrafast")

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["u", "--x264-preset", "warp9"])


if __name__ == "__main__":
    unittest.main()
