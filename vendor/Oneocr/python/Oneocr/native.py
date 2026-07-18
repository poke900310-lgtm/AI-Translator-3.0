"""Python native OCR bridge for the embedded OneOcr runtime.

This build keeps the existing OCR engine but removes avoidable image
serialization in the hot path, reuses the native OCR pipeline across calls, and
supports a lighter line-only mode for rescans and plain-text consumers.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctypes import wintypes

from .common import AppContext, OcrError
from .upscale import convert_to_prepared_image


class ImageStructure(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("reserved_", ctypes.c_int),
        ("step_size", ctypes.c_longlong),
        ("data_ptr", ctypes.c_void_p),
    ]


class PointF(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class BoundingBox(ctypes.Structure):
    _fields_ = [
        ("top_left", PointF),
        ("top_right", PointF),
        ("bottom_right", PointF),
        ("bottom_left", PointF),
    ]


BoundingBoxPtr = ctypes.POINTER(BoundingBox)


@dataclass(slots=True)
class OcrWord:
    text: str
    bounding_box: BoundingBox
    confidence: float | None = None


@dataclass(slots=True)
class OcrLine:
    text: str
    bounding_box: BoundingBox
    words: list[OcrWord]


@dataclass(slots=True)
class OcrResult:
    lines: list[OcrLine]
    image_angle: float = 0.0


class OneOcrEngine:
    """Thin ctypes wrapper around the OneOcr native runtime.

    The native DLL and OCR pipeline are initialized once and then reused across
    frames. Calls are serialized through a lock because the translator may ask
    for small rescans while a background OCR worker is still active.
    """

    def __init__(self, runtime_dir: Path, context: AppContext) -> None:
        self.runtime_dir = runtime_dir
        self.context = context
        self.config = context.config
        self._dll_directory_handle = None
        self._run_lock = threading.RLock()

        self._check_required_files()
        self._configure_dll_search_path()
        self.dll = ctypes.WinDLL(str(self.runtime_dir / self.config["ocr"]["dll_name"]))
        self._bind_functions()

        self._model_path_buf = ctypes.create_string_buffer(self._utf8_z(str(self.runtime_dir / self.config["ocr"]["model_name"])))
        self._key_buf = ctypes.create_string_buffer(self._utf8_z(str(self.config["ocr"]["model_key"])))
        self._init_options = ctypes.c_longlong(0)
        self._process_options = ctypes.c_longlong(0)
        self._pipeline = ctypes.c_longlong(0)
        self._closed = False
        self._ensure_pipeline_ready()

    def _check_required_files(self) -> None:
        required_files = self.config["required_runtime_files"]
        missing = [name for name in required_files if not (self.runtime_dir / name).is_file()]
        if missing:
            raise OcrError(f"Missing required file(s) in runtime folder: {', '.join(missing)}")

    def _configure_dll_search_path(self) -> None:
        if hasattr(os, "add_dll_directory"):
            self._dll_directory_handle = os.add_dll_directory(str(self.runtime_dir))

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_dll_directory = kernel32.SetDllDirectoryW
        set_dll_directory.argtypes = [wintypes.LPCWSTR]
        set_dll_directory.restype = wintypes.BOOL

        if not set_dll_directory(str(self.runtime_dir)):
            raise OcrError(f"SetDllDirectory failed for: {self.runtime_dir}")

    def _bind_functions(self) -> None:
        ll = ctypes.c_longlong
        byte = ctypes.c_ubyte
        pvoid = ctypes.c_void_p

        self.CreateOcrInitOptions = self.dll.CreateOcrInitOptions
        self.CreateOcrInitOptions.argtypes = [ctypes.POINTER(ll)]
        self.CreateOcrInitOptions.restype = ll

        self.OcrInitOptionsSetUseModelDelayLoad = self.dll.OcrInitOptionsSetUseModelDelayLoad
        self.OcrInitOptionsSetUseModelDelayLoad.argtypes = [ll, byte]
        self.OcrInitOptionsSetUseModelDelayLoad.restype = ll

        self.CreateOcrProcessOptions = self.dll.CreateOcrProcessOptions
        self.CreateOcrProcessOptions.argtypes = [ctypes.POINTER(ll)]
        self.CreateOcrProcessOptions.restype = ll

        self.OcrProcessOptionsSetMaxRecognitionLineCount = self.dll.OcrProcessOptionsSetMaxRecognitionLineCount
        self.OcrProcessOptionsSetMaxRecognitionLineCount.argtypes = [ll, ll]
        self.OcrProcessOptionsSetMaxRecognitionLineCount.restype = ll

        self.CreateOcrPipeline = self.dll.CreateOcrPipeline
        self.CreateOcrPipeline.argtypes = [pvoid, pvoid, ll, ctypes.POINTER(ll)]
        self.CreateOcrPipeline.restype = ll

        self.RunOcrPipeline = self.dll.RunOcrPipeline
        self.RunOcrPipeline.argtypes = [ll, ctypes.POINTER(ImageStructure), ll, ctypes.POINTER(ll)]
        self.RunOcrPipeline.restype = ll

        self.GetImageAngle = getattr(self.dll, "GetImageAngle", None)
        if self.GetImageAngle is not None:
            self.GetImageAngle.argtypes = [ll, ctypes.POINTER(ctypes.c_float)]
            self.GetImageAngle.restype = ll

        self.GetOcrLineCount = self.dll.GetOcrLineCount
        self.GetOcrLineCount.argtypes = [ll, ctypes.POINTER(ll)]
        self.GetOcrLineCount.restype = ll

        self.GetOcrLine = self.dll.GetOcrLine
        self.GetOcrLine.argtypes = [ll, ll, ctypes.POINTER(ll)]
        self.GetOcrLine.restype = ll

        self.GetOcrLineContent = self.dll.GetOcrLineContent
        self.GetOcrLineContent.argtypes = [ll, ctypes.POINTER(pvoid)]
        self.GetOcrLineContent.restype = ll

        self.GetOcrLineBoundingBox = getattr(self.dll, "GetOcrLineBoundingBox", None)
        if self.GetOcrLineBoundingBox is not None:
            self.GetOcrLineBoundingBox.argtypes = [ll, ctypes.POINTER(BoundingBoxPtr)]
            self.GetOcrLineBoundingBox.restype = ll

        self.GetOcrLineWordCount = getattr(self.dll, "GetOcrLineWordCount", None)
        if self.GetOcrLineWordCount is not None:
            self.GetOcrLineWordCount.argtypes = [ll, ctypes.POINTER(ll)]
            self.GetOcrLineWordCount.restype = ll

        self.GetOcrWord = getattr(self.dll, "GetOcrWord", None)
        if self.GetOcrWord is not None:
            self.GetOcrWord.argtypes = [ll, ll, ctypes.POINTER(ll)]
            self.GetOcrWord.restype = ll

        self.GetOcrWordContent = getattr(self.dll, "GetOcrWordContent", None)
        if self.GetOcrWordContent is not None:
            self.GetOcrWordContent.argtypes = [ll, ctypes.POINTER(pvoid)]
            self.GetOcrWordContent.restype = ll

        self.GetOcrWordBoundingBox = getattr(self.dll, "GetOcrWordBoundingBox", None)
        if self.GetOcrWordBoundingBox is not None:
            self.GetOcrWordBoundingBox.argtypes = [ll, ctypes.POINTER(BoundingBoxPtr)]
            self.GetOcrWordBoundingBox.restype = ll

        self.GetOcrWordConfidence = getattr(self.dll, "GetOcrWordConfidence", None)
        if self.GetOcrWordConfidence is not None:
            self.GetOcrWordConfidence.argtypes = [ll, ctypes.POINTER(ctypes.c_float)]
            self.GetOcrWordConfidence.restype = ll

        self.ReleaseOcrResult = self.dll.ReleaseOcrResult
        self.ReleaseOcrResult.argtypes = [ll]
        self.ReleaseOcrResult.restype = None

        self.ReleaseOcrInitOptions = self.dll.ReleaseOcrInitOptions
        self.ReleaseOcrInitOptions.argtypes = [ll]
        self.ReleaseOcrInitOptions.restype = None

        self.ReleaseOcrPipeline = self.dll.ReleaseOcrPipeline
        self.ReleaseOcrPipeline.argtypes = [ll]
        self.ReleaseOcrPipeline.restype = None

        self.ReleaseOcrProcessOptions = self.dll.ReleaseOcrProcessOptions
        self.ReleaseOcrProcessOptions.argtypes = [ll]
        self.ReleaseOcrProcessOptions.restype = None

    @staticmethod
    def _check(result_code: int, step: str) -> None:
        if result_code != 0:
            raise OcrError(f"{step} failed with code {result_code}")

    @staticmethod
    def _utf8_z(value: str) -> bytes:
        return value.encode("utf-8") + b"\x00"

    @staticmethod
    def _decode_utf8_ptr(text_ptr: ctypes.c_void_p) -> str:
        if not text_ptr.value:
            return ""
        return ctypes.string_at(text_ptr.value).decode("utf-8")

    def _load_image_as_bgra(self, input_source: Any) -> tuple[bytes, int, int, int]:
        image_config = self.config["image"]
        min_size = int(image_config["min_size"])
        max_size = int(image_config["max_size"])
        upscale_small_images = bool(image_config.get("upscale_small_images", True))

        try:
            return convert_to_prepared_image(
                input_source,
                min_size,
                max_size,
                upscale_small_images,
            )
        except OcrError:
            raise
        except Exception as exc:
            raise OcrError(str(exc)) from exc

    @staticmethod
    def _empty_box() -> BoundingBox:
        return BoundingBox(PointF(0.0, 0.0), PointF(0.0, 0.0), PointF(0.0, 0.0), PointF(0.0, 0.0))

    @staticmethod
    def _copy_box(box: BoundingBox) -> BoundingBox:
        return BoundingBox(
            PointF(float(box.top_left.x), float(box.top_left.y)),
            PointF(float(box.top_right.x), float(box.top_right.y)),
            PointF(float(box.bottom_right.x), float(box.bottom_right.y)),
            PointF(float(box.bottom_left.x), float(box.bottom_left.y)),
        )

    def _get_line_box(self, line_handle: ctypes.c_longlong) -> BoundingBox:
        if self.GetOcrLineBoundingBox is None:
            return self._empty_box()
        bbox_ptr = BoundingBoxPtr()
        if self.GetOcrLineBoundingBox(line_handle, ctypes.byref(bbox_ptr)) != 0 or not bbox_ptr:
            return self._empty_box()
        return self._copy_box(bbox_ptr.contents)

    def _get_word_box(self, word_handle: ctypes.c_longlong) -> BoundingBox:
        if self.GetOcrWordBoundingBox is None:
            return self._empty_box()
        bbox_ptr = BoundingBoxPtr()
        if self.GetOcrWordBoundingBox(word_handle, ctypes.byref(bbox_ptr)) != 0 or not bbox_ptr:
            return self._empty_box()
        return self._copy_box(bbox_ptr.contents)

    def _ensure_pipeline_ready(self) -> None:
        if self._closed:
            raise OcrError('OCR engine is closed.')
        if self._pipeline.value:
            return
        self._check(self.CreateOcrInitOptions(ctypes.byref(self._init_options)), 'CreateOcrInitOptions')
        self._check(self.OcrInitOptionsSetUseModelDelayLoad(self._init_options, 0), 'OcrInitOptionsSetUseModelDelayLoad')
        self._check(
            self.CreateOcrPipeline(
                ctypes.cast(self._model_path_buf, ctypes.c_void_p),
                ctypes.cast(self._key_buf, ctypes.c_void_p),
                self._init_options,
                ctypes.byref(self._pipeline),
            ),
            'CreateOcrPipeline',
        )
        self._check(self.CreateOcrProcessOptions(ctypes.byref(self._process_options)), 'CreateOcrProcessOptions')
        self._check(
            self.OcrProcessOptionsSetMaxRecognitionLineCount(
                self._process_options,
                int(self.config['ocr']['max_recognition_line_count']),
            ),
            'OcrProcessOptionsSetMaxRecognitionLineCount',
        )

    def close(self) -> None:
        with self._run_lock:
            if self._closed:
                return
            self._closed = True
            if self._process_options.value:
                self.ReleaseOcrProcessOptions(self._process_options)
                self._process_options = ctypes.c_longlong(0)
            if self._pipeline.value:
                self.ReleaseOcrPipeline(self._pipeline)
                self._pipeline = ctypes.c_longlong(0)
            if self._init_options.value:
                self.ReleaseOcrInitOptions(self._init_options)
                self._init_options = ctypes.c_longlong(0)
            if self._dll_directory_handle is not None:
                try:
                    self._dll_directory_handle.close()
                except Exception:
                    pass
                self._dll_directory_handle = None

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def recognize_result(self, source_image: Any, include_words: bool = True) -> OcrResult:
        """Run OCR and return structured lines, words, and geometry."""

        with self._run_lock:
            self._ensure_pipeline_ready()
            ll = ctypes.c_longlong
            ocr_result = ll(0)
            image_buf = None
            try:
                image_bytes, width, height, step = self._load_image_as_bgra(source_image)
                image_buf = ctypes.create_string_buffer(image_bytes)
                image = ImageStructure(
                    type=3,
                    width=width,
                    height=height,
                    reserved_=0,
                    step_size=step,
                    data_ptr=ctypes.cast(image_buf, ctypes.c_void_p).value,
                )

                self._check(
                    self.RunOcrPipeline(
                        self._pipeline,
                        ctypes.byref(image),
                        self._process_options,
                        ctypes.byref(ocr_result),
                    ),
                    'RunOcrPipeline',
                )

                angle = 0.0
                if self.GetImageAngle is not None:
                    angle_value = ctypes.c_float(0.0)
                    if self.GetImageAngle(ocr_result, ctypes.byref(angle_value)) == 0:
                        angle = float(angle_value.value)

                line_count = ll(0)
                self._check(self.GetOcrLineCount(ocr_result, ctypes.byref(line_count)), 'GetOcrLineCount')

                lines: list[OcrLine] = []
                for i in range(line_count.value):
                    line_handle = ll(0)
                    if self.GetOcrLine(ocr_result, i, ctypes.byref(line_handle)) != 0:
                        continue

                    text_ptr = ctypes.c_void_p()
                    if self.GetOcrLineContent(line_handle, ctypes.byref(text_ptr)) != 0:
                        continue
                    line_text = self._decode_utf8_ptr(text_ptr).strip()
                    if not line_text:
                        continue

                    words: list[OcrWord] = []
                    if include_words and self.GetOcrLineWordCount is not None and self.GetOcrWord is not None:
                        word_count = ll(0)
                        if self.GetOcrLineWordCount(line_handle, ctypes.byref(word_count)) == 0:
                            for j in range(word_count.value):
                                word_handle = ll(0)
                                if self.GetOcrWord(line_handle, j, ctypes.byref(word_handle)) != 0:
                                    continue
                                word_ptr = ctypes.c_void_p()
                                if self.GetOcrWordContent is None or self.GetOcrWordContent(word_handle, ctypes.byref(word_ptr)) != 0:
                                    continue
                                word_text = self._decode_utf8_ptr(word_ptr).strip()
                                if not word_text:
                                    continue
                                confidence = None
                                if self.GetOcrWordConfidence is not None:
                                    conf = ctypes.c_float(0.0)
                                    if self.GetOcrWordConfidence(word_handle, ctypes.byref(conf)) == 0:
                                        confidence = float(conf.value)
                                words.append(
                                    OcrWord(
                                        text=word_text,
                                        bounding_box=self._get_word_box(word_handle),
                                        confidence=confidence,
                                    )
                                )

                    lines.append(
                        OcrLine(
                            text=line_text,
                            bounding_box=self._get_line_box(line_handle),
                            words=words,
                        )
                    )

                return OcrResult(lines=lines, image_angle=angle)
            finally:
                if ocr_result.value:
                    self.ReleaseOcrResult(ocr_result)

    def recognize_text(self, source_image: Any) -> str:
        result = self.recognize_result(source_image, include_words=False)
        return "\n".join(line.text for line in result.lines)

    def recognize_text_fast(self, source_image: Any) -> str:
        return self.recognize_text(source_image)
