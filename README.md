# AI-Translator 3.0 with embedded One OCR


## OCR integration

This build uses the embedded Python One OCR engine directly inside the
translator process via a ctypes bridge to `oneocr.dll`.

Why an in-process engine:
- lower latency
- fewer process/protocol failure points
- simpler logging and debugging
- more reliable long-running capture sessions

The app is designed around a **single full-window workflow** with optional
user-defined translation zones for finer control.

## Workflow

1. Start the app.
2. Choose a visible target application window.
3. Click **Attach**. Full-window capture starts immediately.
4. The app captures the **full client area** of that window, runs One OCR over
   the whole image (or over user-defined zones), tracks OCR blocks across
   frames, suppresses text that stays unchanged for a long time, translates
   dynamic blocks through the local `llama-server`, and paints the translated
   text back over the original text regions.

There are no alternate capture modes in this build.

## What changed

- full-window capture instead of a user-drawn ROI
- structured OCR with line and word bounding boxes from the bundled OCR runtime
- block tracking across time to separate dynamic text from persistent UI text
- in-place overlay rendering instead of a dark subtitle panel
- local background reconstruction under translated text using blurred source
  pixels plus sampled surrounding colors
- translated text color and outline are sampled from the source text region

## Current file structure

```text
.
├─ app/
│  ├─ __init__.py        # exports VERSION
│  ├─ color.py           # color sampling for fill / outline / background
│  ├─ config.py          # runtime tunables
│  ├─ controller.py      # capture / OCR / track / translate / render orchestrator
│  ├─ geometry.py        # Rect math helpers
│  ├─ llama_server.py    # llama.cpp HTTP client
│  ├─ logging.py         # multi-channel JSON logger + artifact paths
│  ├─ one_ocr.py         # Microsoft OneOCR wrapper
│  ├─ qt_render.py       # Qt overlay paint loop
│  ├─ screen_capture.py  # Win32 client-area capture
│  ├─ text.py            # text normalisation + CJK detection
│  ├─ types.py           # dataclasses (Track, Observation, Rect, etc.)
│  ├─ window_binding.py  # owner / z-order / capture-exclude glue
│  └─ windows.py         # control panel + region/zone overlay windows
├─ docs/
│  └─ PATCH_NOTES_v0.3.28.md
├─ scripts/
│  ├─ generate_debug_composites.py
│  └─ reset_state.py
├─ vendor/
│  └─ Oneocr/
├─ install.ps1
├─ main.py
├─ requirements.txt
└─ run.bat
```

## Main files

- `main.py`
  - starts the app, enables DPI awareness, ensures `llama-server` is running,
    and wires together the Qt control panel, overlay, and controller

- `app/controller.py`
  - central runtime controller
  - captures the full attached window, runs OCR, groups OCR lines into blocks,
    tracks them across frames, suppresses static text, requests translations,
    and builds the in-place overlay scene

- `app/screen_capture.py`
  - captures the **client area of the attached window** through Win32 window
    capture instead of a desktop rectangle

- `app/one_ocr.py`
  - adapter around the embedded One OCR runtime
  - exposes structured OCR results with line and word boxes

- `vendor/Oneocr/python/Oneocr/native.py`
  - low-level ctypes bridge to `oneocr.dll`
  - reads line text, word text, bounding boxes, and confidences from the native
    OCR result

- `app/llama_server.py`
  - sends each dynamic OCR block to the local OpenAI-compatible
    `llama-server` endpoint and returns the translated text

- `app/windows.py`
  - contains the Qt control panel and the transparent full-window overlay that
    redraws translated text in place

- `app/window_binding.py`
  - enumerates visible top-level windows
  - resolves the attached target window's client area in screen coordinates
  - applies capture exclusion to the translation overlay where Windows supports it

- `app/config.py`
  - stores tuning values for capture cadence, track stability, static-text
    detection, patch reconstruction, text rendering, and llama-server settings

## Source language selection

On startup, the app still asks for a source language hint:
- Japanese (`ja`)
- Korean (`ko`)
- Chinese Simplified (`zh-Hans`)
- Chinese Traditional (`zh-Hant`)
- English (`en`)
- Auto / no hint (`0`)

This is used as a **translation hint**, not as an OCR-engine selector.

## Setup (Windows)

1. Put any compatible GGUF model wherever you want.
2. Run `install.ps1`.
3. Make sure `LLAMA_SERVER_MODEL_PATH` in `config.py` or your environment points
   to the model you want.
4. Start the app with `run.bat`.
5. Attach to the app window that contains the source text.

## Notes

- This build is Windows x64 only because the OCR runtime is native.
- The in-place overlay depends on OCR bounding boxes. If a target application
  renders text in a very unusual way, the translation patch may still need
  tuning.
- Static-text suppression is temporal. Text is treated as persistent UI only
  after it stays essentially unchanged for multiple consecutive frames.


## Debugging capture

This build writes `overlay_translator_debug.log` next to the app by default.
If capture fails, the log records the target window title, class name, PID,
visibility/minimized state, and the resolved client rectangle.

The current capture path reads the target window's client area from the
visible desktop compositor instead of using `PrintWindow`. That avoids common
`PrintWindow failed` errors, but it means the target window must be visible on
screen while translating.


## Debug artifacts

The app writes cross-referenceable artifacts to `debug/` next to the app
when enabled. Channel logs always run; image / text snapshots respect
`DEBUG_MODE` in `app/config.py` (see below).

- `debug/overlay_translator_debug.log` and channel logs in `debug/logs/`
- `logs/capture.log` for full-window capture timing, rects, and saved frame paths
- `logs/ocr.log` for raw OCR lines, accepted blocks, boxes, and text snapshot paths
- `logs/tracks.log` for track create/update/static/delete decisions
- `logs/translate.log` for llama request/response timing and saved source/result text
- `logs/render.log` for rendered overlay items and saved preview images
- `images/captures/` raw captured frames
- `images/annotated/` OCR and track box overlays
- `images/rendered/` preview composites of the translated scene
- `text/ocr_raw/`, `text/ocr_blocks/`, `text/translate_source/`, and `text/translate_result/`
- `debug/memory/` per-window translation memory (when `TRANSLATION_MEMORY_ENABLED = True`)

Use `frame_index` plus timestamps to cross-reference the logs.

`DEBUG_MODE` (in `app/config.py`) controls image/text capture volume:

- `"lite"` (default) — channel logs always; image and text artifacts only
  on interesting frames at a low heartbeat rate.
- `"verbose"` — legacy behavior, full per-N sampling.
- `"off"` — channel logs still run, but no PNG / TXT files land on disk.

Key channel events to grep when validating a run:

```
Select-String INPAINT_DECISION       debug/logs/render.log
Select-String TEXT_SETTLED           debug/logs/translate.log
Select-String QUEUE_HOLD_UNSTABLE    debug/logs/translate.log
Select-String ZONE_JITTER_SUPPRESSED debug/logs/tracks.log
```


## Patch notes

Current build version: see `VERSION` in `app/__init__.py` (currently 0.3.28).

Recent passes focus on stable text-change detection, per-zone trigger
isolation, more reliable inpaint-based patch reconstruction, and a lighter
event-driven debug capture pipeline.


## Development

The repo carries `pyproject.toml` configuration for pytest, mypy, and ruff,
plus a baseline test suite under `tests/`. To install dev tooling:

```
.\install.ps1 -Dev               # also installs pytest / mypy / ruff
# or, on an existing venv:
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

To run the quality gates:

```
.\.venv\Scripts\python.exe -m ruff check app tests main.py scripts
.\.venv\Scripts\python.exe -m mypy app/
.\.venv\Scripts\python.exe -m pytest -q
```

The `tests/` suite covers pure-function modules plus regression tests for
the inpaint trigger, text-stability gate, and per-zone trigger fixes.
