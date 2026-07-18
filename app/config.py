"""OverlayTranslator2 runtime configuration.

This build captures the full client area of one attached window, runs the
bundled One OCR engine across that window, classifies OCR blocks as dynamic
or static over time, and paints translated text back over the original text
regions.
"""

import json
import os
from pathlib import Path

_RUNTIME_SETTINGS_PATH = Path(
    os.getenv("AI_TRANSLATE_RUNTIME_CONFIG", str(Path(__file__).resolve().parent.parent / "config.runtime.json"))
)

# Cached parse of config.runtime.json. The original load_runtime_settings()
# parsed the file on every call, which meant every _runtime_int() lookup hit
# the filesystem. We now parse once on demand and invalidate when the file's
# mtime changes (e.g. after reset_state.py or a manual edit).
_RUNTIME_SETTINGS_CACHE: dict[str, object] | None = None
_RUNTIME_SETTINGS_MTIME: float = -1.0


def _read_runtime_settings_from_disk() -> dict[str, object]:
    try:
        if _RUNTIME_SETTINGS_PATH.exists():
            data = json.loads(_RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def load_runtime_settings() -> dict[str, object]:
    global _RUNTIME_SETTINGS_CACHE, _RUNTIME_SETTINGS_MTIME
    try:
        mtime = _RUNTIME_SETTINGS_PATH.stat().st_mtime if _RUNTIME_SETTINGS_PATH.exists() else -1.0
    except Exception:
        mtime = -1.0
    if _RUNTIME_SETTINGS_CACHE is None or mtime != _RUNTIME_SETTINGS_MTIME:
        _RUNTIME_SETTINGS_CACHE = _read_runtime_settings_from_disk()
        _RUNTIME_SETTINGS_MTIME = mtime
    return _RUNTIME_SETTINGS_CACHE


def save_runtime_settings(updates: dict[str, object]) -> None:
    global _RUNTIME_SETTINGS_CACHE, _RUNTIME_SETTINGS_MTIME
    settings = dict(load_runtime_settings())
    settings.update(updates or {})
    _RUNTIME_SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    _RUNTIME_SETTINGS_CACHE = settings
    try:
        _RUNTIME_SETTINGS_MTIME = _RUNTIME_SETTINGS_PATH.stat().st_mtime
    except Exception:
        _RUNTIME_SETTINGS_MTIME = -1.0


def _runtime_int(name: str, default: int) -> int:
    settings = load_runtime_settings()
    raw = settings.get(name, default)
    try:
        # raw is object here (settings is dict[str, object]); narrow via str().
        return int(str(raw)) if not isinstance(raw, int) else raw
    except (TypeError, ValueError):
        return int(default)


def _runtime_float(name: str, default: float) -> float:
    settings = load_runtime_settings()
    raw = settings.get(name, default)
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw))
    except (TypeError, ValueError):
        return float(default)


DEBUG_PROFILE = os.getenv("AI_TRANSLATE_DEBUG_PROFILE", "forensic").strip().lower()
_DEBUG_FORENSIC = DEBUG_PROFILE == "forensic"

# Capture cadence and OCR scheduling in milliseconds.
CAPTURE_INTERVAL_MS = 125
OCR_INTERVAL_MS = 250
OCR_MIN_INTERVAL_MS = 250

# Capture backend policy.
# Keep the fast desktop BitBlt path, but freeze capture when foreign windows
# cover the target so OCR never sees the wrong content.
CAPTURE_INCLUDE_LAYERED_WINDOWS = False
CAPTURE_PAUSE_WHEN_OCCLUDED = True
CAPTURE_OCCLUSION_SAMPLE_COLS = 3
CAPTURE_OCCLUSION_SAMPLE_ROWS = 3
CAPTURE_OCCLUSION_INSET_PX = 12

# Runtime-configurable OCR/render inset around the hooked window edge.
EDGE_IGNORE_PADDING = _runtime_int("EDGE_IGNORE_PADDING", 1)

# Translation stabilization and track persistence.
TRACK_STABLE_FRAMES = 2
# Per-classification stable-frame thresholds. Previously the dialogue/name/sign
# fast paths dropped this to 1 frame, which let a single-frame OCR snapshot
# queue translation against partially-rendered text. 2 is the minimum that
# rejects single-frame jitter while still feeling responsive on dialog.
TRACK_STABLE_FRAMES_DIALOGUE = 2
TRACK_STABLE_FRAMES_SIGN = 2
TRACK_STABLE_FRAMES_NAME = 2
# How many consecutive frames a track can be "changing" before we explicitly
# suppress its stale translation. After this many unstable frames we clear
# the prior translation and emit QUEUE_HOLD_UNSTABLE instead of QUEUE_SKIP,
# so the overlay doesn't flash an old result while text is in motion.
TEXT_UNSTABLE_HOLD_FRAMES = 4
# Type-on dialogue reveal hold. When OCR text is strictly growing
# (new frame's text starts with the prior text, longer than before), the
# track is in a "still being revealed" state. After REVEAL_HOLD_STREAK_FRAMES
# consecutive growth frames _maybe_request_translation skips with
# reason="text_revealing" instead of queuing a partial-text translation.
# A short pause (a frame without growth) resets the streak.
REVEAL_HOLD_STREAK_FRAMES = 3
# Minimum new/old length ratio that counts as "growth" rather than a complete
# text replacement. >= 1.0 means strictly longer; tolerate small OCR jitter.
REVEAL_HOLD_MIN_GROWTH_RATIO = 1.02
TRACK_FORGET_FRAMES = 8
MATCH_IOU_THRESHOLD = 0.12
MATCH_DISTANCE_PX = 120
MATCH_TEXT_REUSE_MAX_DISTANCE_PX = 84
MATCH_TEXT_REUSE_MAX_DISTANCE_RATIO = 0.85
MATCH_TEXT_REUSE_IOU_FLOOR = 0.08
MATCH_TEXT_REUSE_PENALTY = 0.42
REPLACEMENT_OBS_IOU_MIN = 0.28
REPLACEMENT_OBS_TEXT_MAX_RATIO = 0.74
REPLACEMENT_OBS_HORIZONTAL_OVERLAP_MIN = 0.55
REPLACEMENT_OBS_VERTICAL_OVERLAP_MIN = 0.55

# Rendering. Patches are blur+solid blends of the underlying scene tinted
# toward the sampled background colour. cv2.inpaint was removed because it
# produced visible glyph-remnant artifacts on busy scenes.
# All the padding/margin defaults are now 0. Their original values were
# chosen back when the patch was a fixed rounded-rectangle behind the
# translation, so the padding was cosmetic breathing room around the
# glyphs. In practice it makes the rendered overlay noticeably larger
# than the source it's covering, which the user consistently tunes back
# down. Kept as runtime knobs in case a specific scene needs them.
PATCH_EXPAND_X = _runtime_int("PATCH_EXPAND_X", 0)
PATCH_EXPAND_Y = _runtime_int("PATCH_EXPAND_Y", 0)
PATCH_BLUR_RADIUS = 7
PATCH_SOLID_BLEND = 0.34
PATCH_EDGE_FEATHER = 3
TEXT_MARGIN_X = _runtime_int("TEXT_MARGIN_X", 0)
TEXT_MARGIN_Y = _runtime_int("TEXT_MARGIN_Y", 0)
TEXT_MIN_POINT = 10
TEXT_MAX_POINT = 28
TEXT_MIN_PIXEL = _runtime_int("TEXT_MIN_PIXEL", 10)
TEXT_MAX_PIXEL = _runtime_int("TEXT_MAX_PIXEL", 26)
TEXT_SOURCE_HEIGHT_SCALE = _runtime_float("TEXT_SOURCE_HEIGHT_SCALE", 0.7)
TEXT_FONT_FAMILY = "Segoe UI"
TEXT_FONT_WEIGHT = 600
TEXT_OUTLINE_WIDTH = 1
# Default rendered-text alignment when nothing more specific is requested.
# Accepted values: "top_left" (default), "top_right", "left" (legacy
# left+vertical-center), "center". "top_left" anchors the translated text
# at the upper-left of the patch so it follows reading order and doesn't
# sit on top of the source row's vertical center during cross-frame
# transitions.
TEXT_DEFAULT_ALIGNMENT = "top_left"
# Cap how aggressively _resolve_layout can shrink the font when text
# doesn't fit. With the source-derived preferred size, a 26px source
# should not silently render at 10px just because the translation grew
# slightly. Lower ratio = more uniform across boxes but more clipping
# risk on dense translations.
TEXT_MAX_SHRINK_RATIO = 0.85
# Outline radius scales with pixel size so large text gets a proportional
# stroke instead of a thin 1px halo. Effective radius =
# max(TEXT_OUTLINE_WIDTH, round(pixel_size / TEXT_OUTLINE_SCALE_DIVISOR)).
TEXT_OUTLINE_SCALE_DIVISOR = 14
THIN_LABEL_WRAP_RATIO = 1.75

# Change-detection and render retention.
OCR_QUEUE_LATEST_ONLY = True
PRESENTATION_STRICT_MONOTONIC = True
PIXEL_CHANGE_SIGNATURE_SIZE = 24
PIXEL_CHANGE_TEXT_SIGNATURE_SIZE = 32
PIXEL_CHANGE_WORD_EXPAND_X = 2
PIXEL_CHANGE_WORD_EXPAND_Y = 2
PIXEL_CHANGE_EXPAND_X = 6
PIXEL_CHANGE_EXPAND_Y = 4
# Region-change detection is fraction-based, not mean-delta based. A small
# pulsing effect (cursor, sparkle, status dot) only modifies a handful of
# downscaled signature cells — its mean-delta contribution is tiny but its
# absolute "I changed something" signal would still trigger a fixed-delta
# gate over time. Switching to "fraction of cells that changed by at least
# CELL_DELTA_MIN" means: small effects on a long text region don't hide the
# overlay, only large-fraction changes do.
#
# Each "cell" is a downscaled pixel of the 24×24 context + 32×32 detail +
# 32×32 edge signature (2624 cells total). CELL_DELTA_MIN is the per-cell
# brightness change (0-255) that counts as "this cell changed".

# OCR re-trigger gate — cheaper false-positive cost (just an extra OCR pass)
# so this stays more sensitive.
PIXEL_CHANGE_CELL_DELTA_MIN = 16
PIXEL_CHANGE_FRACTION_THRESHOLD = 0.01  # 1% of cells must change to retrigger
PIXEL_CHANGE_DELETE_AFTER_FRAMES = 2

# Periodic full-rescan safety net. Even when the pixel-change gate decides
# nothing moved, force a clean OCR pass at this cadence so missed
# translations get a second chance. Pixel/zone signatures are reset on the
# rescan tick so the next frame's gate sees everything as "changed".
FORCE_FULL_RESCAN_INTERVAL_MS = 1000

# Fast-liveness render-hide gate — false positives manifest as visible
# flicker, so this is strictly more permissive than the OCR path.
LIVENESS_CELL_DELTA_MIN = 24
LIVENESS_CHANGED_FRACTION_THRESHOLD = 0.20  # 20% of cells must change to hide
LIVENESS_PIXEL_CHANGE_DELETE_AFTER_FRAMES = 4

# Catastrophic-change fallback. Mean abs delta across all cells (0-255 scale).
# Even if the fraction gate misses (e.g. a global brightness shift where
# every cell changes only ~10 units), this catches the big-cut case so the
# overlay still hides on a real scene change.
PIXEL_CHANGE_HARD_CUT_THRESHOLD = 48.0

# When True, the liveness signature ignores the wider context band and uses
# only the word-rect detail + edge bytes. Effects outside the actual text
# glyphs (pulsing balls, scrolling sparkles) won't count.
LIVENESS_WORD_FOCUSED_SIGNATURE = True
PIXEL_HOLD_MAX_FRAMES = 240
OCR_DISAPPEAR_CONFIRM_FRAMES = 10
OCR_REPLACED_CONFIRM_FRAMES = 2
FAST_RENDER_LIVENESS_ENABLED = True
FAST_RENDER_LIVENESS_MAX_TRACKS = 24
STALE_RENDER_MAX_MISSING_FRAMES = 3
STALE_RENDER_MAX_MISSING_FRAMES_LOW_VALUE = 2
STALE_RENDER_MAX_MISSING_FRAMES_UI = 1
STALE_RENDER_MAX_MISSING_FRAMES_DIALOGUE = 1
SIZE_CHANGE_FLUSH_ENABLED = True

# HUD/menu detection so persistent UI stays original.
HUD_EDGE_MARGIN_PX = 72
HUD_THIN_MAX_HEIGHT = 24
HUD_BOTTOM_BAND_PX = 32

# Timing diagnostics.
TIMING_LOGGING = True

# Translation caching / memory.
TRANSLATION_MEMORY_ENABLED = True
TRANSLATION_MEMORY_PERSIST = True
TRANSLATION_MEMORY_DIR = "memory"
# Minimum gap between debounced saves. The previous behaviour wrote the
# entire JSON file on every successful translation, which on a long
# session with hundreds of cached entries flooded the disk. With this
# debounce, multiple back-to-back translations coalesce into a single
# write; force=True bypasses the debounce on shutdown / reset.
TRANSLATION_MEMORY_SAVE_INTERVAL_MS = 2000

# Translation backend.
TRANSLATION_BACKEND = "llama_server"
LLAMA_SERVER_BASE_URL = os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080")
LLAMA_SERVER_MODEL = os.getenv("LLAMA_SERVER_MODEL", "local-model")
LLAMA_SERVER_MODEL_PATH = os.getenv("LLAMA_SERVER_MODEL_PATH", r".\models\Sugoi-14B-Ultra-Q4_K_M.gguf")
LLAMA_SERVER_CTX = int(os.getenv("LLAMA_SERVER_CTX", "2048"))
LLAMA_SERVER_NGL = os.getenv("LLAMA_SERVER_NGL", "99")
LLAMA_SERVER_FA = os.getenv("LLAMA_SERVER_FA", "on")
LLAMA_SERVER_TIMEOUT_S = int(os.getenv("LLAMA_SERVER_TIMEOUT_S", "60"))
LLAMA_SERVER_MAX_TOKENS = int(os.getenv("LLAMA_SERVER_MAX_TOKENS", "512"))
# Rolling conversational-context window fed to the translation prompt as
# few-shot (source, translation) pairs so the LLM can resolve pronouns,
# dropped subjects, and tone against what was just said. 0 disables and
# reverts to stateless per-line translation.
TRANSLATION_CONTEXT_MAX_LINES = int(os.getenv("TRANSLATION_CONTEXT_MAX_LINES", "4"))
# Number of recent per-zone char-height samples the render baseline takes
# a running median over. Larger = steadier font size across dialogue lines
# in the same zone but slower to react if the game genuinely changes its
# text size. 0 disables and falls back to per-frame OCR measurement.
ZONE_FONT_BASELINE_WINDOW = int(os.getenv("ZONE_FONT_BASELINE_WINDOW", "10"))

# Logging and debug artifacts.
# Enabled by default in this debug-oriented build so capture/OCR/translation
# failures leave useful artifacts next to the app.
DEBUG_LOG = True
DEBUG_LOG_PATH = "debug/overlay_translator_debug.log"
DEBUG_LOG_DIR = "."

# Top-level debug mode. Three settings:
#   "lite"    — default; image and text artifacts are written only on
#               interesting frames and at a much lower heartbeat rate than
#               the legacy build. Channel logs (capture/ocr/tracks/translate/
#               render) are unaffected — they remain detailed.
#   "verbose" — legacy behavior; honors the DEBUG_SAVE_*_EVERY_N rates below
#               literally. Use for forensic deep-dives.
#   "off"     — skip every image / text artifact write. Channel logs still
#               emit but no PNG / TXT files land on disk.
DEBUG_MODE = "lite"

# Save debug artifacts. The EVERY_N rates below now act as MAXIMUM cadence
# in "lite" mode (i.e. a save will not happen more often than every N
# frames; it can still be skipped by the interesting-frames filter).
# Default cadences were dropped from 4 frames/sample → 32 frames/sample
# (capture / render) and 2 → 24 (annotated / manifest) so that an idle
# session writes ~1 image per ~4-30 seconds at the 8 fps capture cadence
# instead of multiple per second.
DEBUG_SAVE_CAPTURE_IMAGES = True
DEBUG_SAVE_CAPTURE_EVERY_N = 32
DEBUG_SAVE_ANNOTATED_IMAGES = True
DEBUG_ANNOTATED_SHOW_OBSERVATIONS = True
DEBUG_ANNOTATED_SHOW_STALE_TRACKS = False
DEBUG_ANNOTATED_ONLY_RENDERED_TRACKS = False
DEBUG_SAVE_RENDER_PREVIEWS = True
DEBUG_SAVE_RENDER_EVERY_N = 32
DEBUG_SAVE_TEXT_SNAPSHOTS = True
DEBUG_MAX_TEXT_PREVIEW = 400

# Reduce image spam while preserving useful forensics.
DEBUG_SAVE_ONLY_INTERESTING_FRAMES = True
DEBUG_SAVE_HEARTBEAT_EVERY_N = 240
DEBUG_SAVE_CAPTURE_ON_CHANGED_ONLY = True
DEBUG_SAVE_ANNOTATED_EVERY_N = 24
DEBUG_SAVE_RENDER_ONLY_WITH_ITEMS = True
DEBUG_SAVE_MANIFESTS_ONLY_INTERESTING = True

# Cross-reference manifests and ID overlays for forensic debugging
DEBUG_SAVE_FRAME_MANIFESTS = True
DEBUG_FRAME_MANIFEST_EVERY_N = 32
DEBUG_OVERLAY_DEBUG_LABELS = False
DEBUG_LOGGER_QUEUE_SIZE = 16384
DEBUG_LOGGER_CLOSE_TIMEOUT_S = 15.0

# OCR text filtering.
OCR_FILTER_ENABLED = True
MIN_CJK_CHARS = 1
MIN_TEXT_LEN = 2
MAX_PUNCT_RATIO = 0.6
OCR_SHORT_TEXT_MAX_LEN = 12
OCR_REJECT_LEADING_PUNCT_SHORT = True
OCR_REJECT_MIXED_SHORT_LATIN_CJK = True
OCR_MIXED_TAIL_MAX_CONFIDENCE = 0.93
OCR_MIXED_ANY_MAX_CONFIDENCE = 0.97
OCR_MIN_LINE_AVG_CONFIDENCE = 0.58
OCR_MIN_LINE_RECT_AREA = 20
OCR_MIN_WORD_RECT_AREA = 6
OCR_MIN_WORD_DIM_PX = 2
OCR_TINY_EDGE_MARGIN_PX = 24
OCR_TINY_EDGE_MAX_AREA = 40
OCR_TINY_EDGE_MAX_CHARS = 4
OCR_TINY_EDGE_MAX_CONFIDENCE = 0.90
SMALL_TEXT_RESCAN_MAX_CONFIDENCE = 0.78

# Low-value scene-label suppression.
LOW_VALUE_STATIC_FRAME_THRESHOLD = 2
LOW_VALUE_DUPLICATE_MAX_CHARS = 18
LOW_VALUE_DUPLICATE_MAX_HEIGHT = 30
LOW_VALUE_DUPLICATE_MAX_AREA_RATIO = 0.04
LOW_VALUE_SCENE_MAX_BOTTOM_RATIO = 0.82
LOW_VALUE_EDGE_MENU_MIN_LINES = 3
LOW_VALUE_EDGE_MENU_MAX_CHARS = 64
LOW_VALUE_EDGE_MENU_MAX_WIDTH_RATIO = 0.42
LOW_VALUE_HOLD_MAX_FRAMES = 6
LOW_VALUE_DISAPPEAR_CONFIRM_FRAMES = 3
LOW_VALUE_REPLACED_CONFIRM_FRAMES = 1

# Dialogue fast-path for perceived latency.
DIALOGUE_FAST_PATH_ENABLED = True
DIALOGUE_PRIORITY_MIN_TOP_RATIO = 0.35
DIALOGUE_PRIORITY_MIN_WIDTH_RATIO = 0.18
DIALOGUE_PRIORITY_MIN_LINES = 2
DIALOGUE_PRIORITY_MIN_CHARS = 8
DIALOGUE_BOX_MIN_TOP_RATIO = 0.78
DIALOGUE_BOX_MAX_BOTTOM_MARGIN_PX = 88
DIALOGUE_BOX_MIN_SINGLELINE_CHARS = 16
DIALOGUE_BOX_MIN_CONTRAST = 72.0
DIALOGUE_BOX_INLINE_NAME_MAX_CHARS = 8
DIALOGUE_BOX_CHAIN_MIN_COMPACT_CHARS = 20
DIALOGUE_BOX_CHAIN_GAP_MAX_PX = _runtime_int("DIALOGUE_BOX_CHAIN_GAP_MAX_PX", 34)
DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX = _runtime_int("DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX", 132)
# OCR line-grouping thresholds — control whether adjacent OCR rows get
# fused into one Observation. Previously hard-coded as ``max(18, 0.8 *
# max_h)`` in ``_records_should_merge``, which was tuned for tight CJK
# paragraph leading and aggressively fused menu lists into one 10-line
# block. Defaults are now ``max(4, 0.35 * max_h)`` — merges wrapped
# paragraph lines (typical leading ~0.2-0.3 of line height) while leaving
# menu items with visible spacing separated. Setting both to 0 forces
# every OCR row into its own Observation.
OCR_LINE_MERGE_MIN_GAP_PX = _runtime_int("OCR_LINE_MERGE_MIN_GAP_PX", 4)
OCR_LINE_MERGE_LINE_HEIGHT_FACTOR = _runtime_float("OCR_LINE_MERGE_LINE_HEIGHT_FACTOR", 0.35)
# Minimum horizontal-overlap ratio for two OCR rows to be considered a
# continuation of the same column. Was previously ``0.22`` with a
# center-distance fallback that let non-overlapping columns merge; the
# fallback is gone and the overlap floor is now the sole alignment
# signal. Setting to 0.5 requires half of the smaller box to overlap
# horizontally — passes wrapped paragraph text, rejects cross-column
# merges. Set lower to be more permissive with unusual layouts.
OCR_LINE_MERGE_OVERLAP_MIN = _runtime_float("OCR_LINE_MERGE_OVERLAP_MIN", 0.5)

# Scene-level uniformity snap for visually-similar UI tracks. OCR
# sampling produces slightly different fill colors (215 vs 218 vs 206
# vs 214 for the same-looking menu items) and slightly different rect
# heights (17 vs 18 vs 19). ``SCENE_UNIFORMITY_ENABLE=True`` post-
# processes each frame's tracks and snaps clusters of visually-similar
# same-class tracks (delta-E of fill color <= threshold) to their
# median color and median char height. Legit outliers (a highlighted
# menu item in a different colour, a title row at 25px) stay separate
# because their delta-E to the cluster exceeds the threshold.
SCENE_UNIFORMITY_ENABLE = bool(_runtime_int("SCENE_UNIFORMITY_ENABLE", 1))
SCENE_UNIFORMITY_MIN_CLUSTER = _runtime_int("SCENE_UNIFORMITY_MIN_CLUSTER", 3)
SCENE_UNIFORMITY_COLOR_DELTA_E = _runtime_float("SCENE_UNIFORMITY_COLOR_DELTA_E", 25.0)
DIALOGUE_BOX_RENDER_MERGE = True
DIALOGUE_BOX_RENDER_MERGE_MAX_TRACKS = 4
DIALOGUE_BOX_RENDER_MERGE_FRAME_DELTA = 16
DIALOGUE_BOX_SAME_ROW_OVERLAP_MIN = 0.5
DIALOGUE_BOX_SAME_ROW_BASELINE_TOLERANCE_PX = 8
DIALOGUE_BOX_SAME_ROW_TOP_TOLERANCE_PX = 6
DIALOGUE_BOX_SAME_ROW_GAP_MAX_PX = 140
DIALOGUE_ZONE_PURGE_ENABLED = True
DIALOGUE_ZONE_PURGE_MIN_DELTA = 14.0

# Lightweight structured logging for faster live translation.
DEBUG_LOG_OCR_OBSERVATIONS = True
DEBUG_LOG_RENDER_ITEMS = True

# Translation queue shaping for lower perceived latency.
TRANSLATION_LOW_PRIORITY_BACKLOG_LIMIT = 2
TRANSLATION_DEDUPLICATE_PENDING = True
TRANSLATION_DIALOGUE_PRIORITY = 0
TRANSLATION_NORMAL_PRIORITY = 50
TRANSLATION_LOW_PRIORITY = 100

# Region-level memoization for visually unchanged text regions.
REGION_MEMO_ENABLED = True
REGION_MEMO_MAX_ENTRIES = 512
REGION_MEMO_RECT_QUANTUM = 12
REGION_MEMO_MIN_TEXT_SIMILARITY = 0.55

# UI / scene-state separation
TRANSLATE_UI_TEXT = True
UI_SLOT_MIN_LINES = 2
UI_SLOT_MAX_CHARS = 120
UI_SLOT_MAX_TOP_RATIO = 0.86
UI_SLOT_MAX_LEFT_RATIO = 0.26
UI_SLOT_MAX_WIDTH_RATIO = 0.82

# Match-class penalties
MATCH_DIALOGUE_CLASS_MIN_TEXT_RATIO = 0.93
MATCH_UI_CLASS_MIN_TEXT_RATIO = 0.90
MATCH_DIALOGUE_CLASS_PENALTY = 0.30
MATCH_UI_CLASS_PENALTY = 0.55

# Region memo gating
REGION_MEMO_ALLOW_UI = False

# UI / timestamp classification
UI_HOLD_MAX_FRAMES = 4
UI_DISAPPEAR_CONFIRM_FRAMES = 2
UI_REPLACED_CONFIRM_FRAMES = 1
REGION_MEMO_REQUIRE_DIALOGUE_CLASS_MATCH = True

# Signs should still translate; UI should not

# UI language selection and text rendering
TEXT_DIALOGUE_FONT_FAMILY = "Segoe UI"
TEXT_DIALOGUE_FONT_WEIGHT = 500
TEXT_NAME_FONT_FAMILY = "Segoe UI"
TEXT_NAME_FONT_WEIGHT = 600
TEXT_SIGN_FONT_FAMILY = "Segoe UI"
TEXT_SIGN_FONT_WEIGHT = 600
TEXT_NAME_SCALE = 0.92
TEXT_LAYOUT_WRAP_EXTRA_PAD_PX = 6
# Max height ratio for general (non-dialogue, non-name) text_rect growth
# in ``_planned_text_rect``. Was hard-coded as ``1.75`` — now runtime-
# tunable so a user cranking ``TEXT_SOURCE_HEIGHT_SCALE`` above ~1.4
# has an explicit knob to grant the extra vertical room instead of the
# text spilling below the patch or the render zone auto-filling to the
# whole translation zone.
TEXT_LAYOUT_MAX_HEIGHT_RATIO = _runtime_float("TEXT_LAYOUT_MAX_HEIGHT_RATIO", 1.75)
# Max width-growth ratio for ``_planned_text_rect``. 1.0 = never widen;
# text always wraps within the source rect's own width. Bump above 1.0
# to let a long translation spill sideways up to that ratio × base
# width before falling back to row-wrap. Previously the widening was
# capped only by the render bounds (the whole translation zone in
# grouped mode), which made single-line translations render at the far
# corners of the group box even when the source text hugged one edge.
TEXT_LAYOUT_WIDEN_MAX_RATIO = _runtime_float("TEXT_LAYOUT_WIDEN_MAX_RATIO", 1.0)
TEXT_DIALOGUE_EXTRA_MARGIN_X = 14
TEXT_DIALOGUE_EXTRA_MARGIN_Y = 12
TEXT_DIALOGUE_MAX_HEIGHT_RATIO = 3.25
TEXT_NAME_MAX_HEIGHT_RATIO = 2.1
NAME_MAX_CHARS = 16
NAME_MIN_TOP_RATIO = 0.55
NAME_MAX_WIDTH_RATIO = 0.35

# Small sign OCR assist and render padding
SMALL_TEXT_RESCAN_ENABLED = True
SMALL_TEXT_RESCAN_MAX_HEIGHT = 50
SMALL_TEXT_RESCAN_MAX_PER_FRAME = 2
SMALL_TEXT_TARGET_HEIGHT = 80
SMALL_TEXT_PADDING_X = 16
SMALL_TEXT_PADDING_Y = 12
SMALL_TEXT_EDGE_MARGIN = 36
SMALL_TEXT_RESCAN_MIN_SCORE_GAIN = 2
SMALL_SIGN_TEXT_SCALE = 0.98

# Signs may use region memo; UI may not

# Translation fast paths
TRANSLATION_NAME_PRIORITY = 10
TRANSLATION_SIGN_PRIORITY = 20
SIGN_FAST_PATH_ENABLED = True

# Dialogue/name disappearance tuning
DIALOGUE_HOLD_MAX_FRAMES = 3
DIALOGUE_DISAPPEAR_CONFIRM_FRAMES = 1
DIALOGUE_REPLACED_CONFIRM_FRAMES = 1

# Debug/runtime balance
if _DEBUG_FORENSIC:
    DEBUG_SAVE_RENDER_EVERY_N = 1

# Small sign / edge render tuning
SMALL_SIGN_MAX_EXTRA_PX = 1
SMALL_SIGN_MAX_PIXEL = 16

# Rescan / queue / OCR junk tuning
SMALL_TEXT_RESCAN_MAX_TOTAL_MS = 120.0
OCR_REJECT_UPPER_SUFFIX_JUNK = True
OCR_UPPER_SUFFIX_MAX_CONFIDENCE = 0.96
SIGN_FORCE_QUEUE_STABLE_FRAMES = 3
TRANSLATION_SIGN_BACKLOG_LIMIT = 4

# OCR recovery / transition handling
OCR_FORCE_ON_CHANGED_RECENT_DIALOGUE = True
OCR_FORCE_ON_CHANGED_DIALOGUE_LOOKBACK_FRAMES = 36
OCR_UI_ONLY_CONFIRMATION_SCANS = 2
OCR_UI_ONLY_CONFIRMATION_INTERVAL_FRAMES = 6
OCR_TRANSLATION_DROUGHT_FORCE_RECHECK = True
OCR_TRANSLATION_DROUGHT_FRAMES = 18
OCR_TRANSLATION_DROUGHT_INTERVAL_FRAMES = 10
TRACK_AGE_SUPPRESSED_ON_NON_OCR_FRAMES = True

# Color-aware OCR regioning and repeated-label consensus
OCR_COLOR_GROUP_ENABLE = True
OCR_COLOR_GROUP_GAP_FACTOR = 0.68
OCR_COLOR_GROUP_HUGE_GAP_FACTOR = 1.15
OCR_COLOR_GROUP_VERTICAL_OVERLAP_MIN = 0.55
OCR_COLOR_GROUP_SEAM_DELTA_E_MAX = 12.0
OCR_COLOR_GROUP_WIDE_GAP_FACTOR = 0.75
OCR_COLOR_BG_DELTA_E_MAX = 14.0
MERGE_MAX_CENTER_DIST_PX = 300
RENDER_STABLE_FRAMES_REQUIRED = 2
OCR_COLOR_FG_DELTA_E_MAX = 18.0

REPEAT_LABEL_CONSENSUS_ENABLED = True
REPEAT_LABEL_MIN_OBS = 3
REPEAT_LABEL_MIN_STRONG_SUPPORT = 2
REPEAT_LABEL_MIN_TEXT_SIMILARITY = 0.60
REPEAT_LABEL_MAX_LENGTH_DELTA = 2
REPEAT_LABEL_EDGE_PREFIX_RATIO = 0.6
REPEAT_LABEL_REPAIR_MAX_CONFIDENCE = 0.98
REPEAT_LABEL_MAX_CHARS = 24
REPEAT_LABEL_MAX_HEIGHT = 72
REPEAT_LABEL_MAX_HEIGHT_RATIO = 1.45
REPEAT_LABEL_MAX_WIDTH_RATIO = 2.6
REPEAT_LABEL_BG_DELTA_E_MAX = 28.0
REPEAT_LABEL_FG_DELTA_E_MAX = 36.0
REPEAT_LABEL_ROW_OVERLAP_MIN = 0.52
REPEAT_LABEL_COL_OVERLAP_MIN = 0.32

REPEAT_LABEL_MIN_FUZZY_SUPPORT = 3
REPEAT_LABEL_MAX_VERTICAL_SPLIT_LINES = 4
REPEAT_LABEL_VERTICAL_SEAM_DELTA_E_MAX = 10.0

DIALOGUE_CACHE_QUARANTINE_ENABLED = True

# All-purpose pipeline feature flags.
ENABLE_UI_HINTS = True
ENABLE_DIALOGUE_HINTS = True
ENABLE_NAME_HINTS = True
TRANSLATION_SKIP_LOW_VALUE = False
RENDER_VERTICAL_TEXT = True

# Global hotkey to toggle pause/resume without leaving the game window.
# Set to an empty string "" to disable.
HOTKEY_PAUSE_RESUME: str = "F9"
