"""Qt windows for the full-window in-place translation workflow."""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from app import VERSION, config
from app.qt_render import paint_overlay_items, pil_to_qimage
from app.window_binding import (
    apply_exclude_from_capture,
    is_target_foreground,
    list_candidate_windows,
    set_owner_window,
    set_window_topmost,
    stack_window_above_target,
)


def _d_int(value: object, default: int = 0) -> int:
    """Narrow `object` (from a `dict[str, object]` lookup) to int.

    The various zone/ignore-region dicts moving through the Qt layer are typed
    as ``dict[str, object]`` so they can carry mixed payloads (rect, group_all,
    label). When we pull an int field out, ``int(value)`` doesn't type-check
    because ``int`` has no overload for ``object``. This helper isinstance-
    narrows first and falls back to the default on anything weird.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


class MoveHandle(QtWidgets.QLabel):
    """Small draggable handle used to move a frameless utility window."""

    def __init__(self, parent: QtWidgets.QWidget, *, tooltip: str = "Move"):
        super().__init__(parent)
        self.setToolTip(tooltip)
        self.setText("⠿")
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(22, 22)
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        self.setStyleSheet("QLabel { color: rgba(255,255,255,230); background: rgba(0,0,0,120); border-radius: 4px; }")
        self._drag_offset = QtCore.QPoint()

    def mousePressEvent(self, e: QtGui.QMouseEvent):  # type: ignore[override]
        if e.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        parent = self.parentWidget()
        if parent is None:
            return
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        self._drag_offset = e.globalPosition().toPoint() - parent.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):  # type: ignore[override]
        if not (e.buttons() & QtCore.Qt.MouseButton.LeftButton):
            return
        parent = self.parentWidget()
        if parent is None:
            return
        parent.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):  # type: ignore[override]
        self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)


class WindowPickerCombo(QtWidgets.QComboBox):
    opened = QtCore.pyqtSignal()

    def showPopup(self):
        self.opened.emit()
        super().showPopup()


class SettingsDialog(QtWidgets.QDialog):
    settingsApplied = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("OverlayTranslator2 Settings")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        self._saved = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        # Region
        region_box = QtWidgets.QGroupBox("Region", self)
        region_form = QtWidgets.QFormLayout(region_box)
        self._edge_padding = self._int_spin(0, 400, 2, int(getattr(config, "EDGE_IGNORE_PADDING", 30)))
        region_form.addRow("Edge ignore padding", self._edge_padding)
        outer.addWidget(region_box)

        # Patch (background box drawn on top of source text)
        patch_box = QtWidgets.QGroupBox("Patch padding (background box around source text)", self)
        patch_form = QtWidgets.QFormLayout(patch_box)
        self._patch_expand_x = self._int_spin(0, 64, 1, int(getattr(config, "PATCH_EXPAND_X", 0)))
        self._patch_expand_y = self._int_spin(0, 64, 1, int(getattr(config, "PATCH_EXPAND_Y", 0)))
        patch_form.addRow("Patch expand X", self._patch_expand_x)
        patch_form.addRow("Patch expand Y", self._patch_expand_y)
        outer.addWidget(patch_box)

        # Text margin (space inside the patch before the rendered text starts)
        margin_box = QtWidgets.QGroupBox("Text margin (space inside patch before text)", self)
        margin_form = QtWidgets.QFormLayout(margin_box)
        self._text_margin_x = self._int_spin(0, 64, 1, int(getattr(config, "TEXT_MARGIN_X", 0)))
        self._text_margin_y = self._int_spin(0, 64, 1, int(getattr(config, "TEXT_MARGIN_Y", 0)))
        margin_form.addRow("Text margin X", self._text_margin_x)
        margin_form.addRow("Text margin Y", self._text_margin_y)
        outer.addWidget(margin_box)

        # Font sizing
        font_box = QtWidgets.QGroupBox("Translated text font size", self)
        font_form = QtWidgets.QFormLayout(font_box)
        self._text_min_pixel = self._int_spin(6, 60, 1, int(getattr(config, "TEXT_MIN_PIXEL", 10)))
        self._text_max_pixel = self._int_spin(8, 96, 1, int(getattr(config, "TEXT_MAX_PIXEL", 26)))
        self._source_height_scale = QtWidgets.QDoubleSpinBox(self)
        self._source_height_scale.setDecimals(2)
        self._source_height_scale.setRange(0.30, 3.00)
        self._source_height_scale.setSingleStep(0.05)
        self._source_height_scale.setValue(float(getattr(config, "TEXT_SOURCE_HEIGHT_SCALE", 0.7)))
        self._text_layout_max_height = QtWidgets.QDoubleSpinBox(self)
        self._text_layout_max_height.setDecimals(2)
        self._text_layout_max_height.setRange(1.00, 8.00)
        self._text_layout_max_height.setSingleStep(0.05)
        self._text_layout_max_height.setValue(float(getattr(config, "TEXT_LAYOUT_MAX_HEIGHT_RATIO", 1.75)))
        self._text_layout_widen = QtWidgets.QDoubleSpinBox(self)
        self._text_layout_widen.setDecimals(2)
        self._text_layout_widen.setRange(1.00, 5.00)
        self._text_layout_widen.setSingleStep(0.05)
        self._text_layout_widen.setValue(float(getattr(config, "TEXT_LAYOUT_WIDEN_MAX_RATIO", 1.0)))
        font_form.addRow("Minimum pixel size", self._text_min_pixel)
        font_form.addRow("Maximum pixel size", self._text_max_pixel)
        font_form.addRow("Scale of source char height", self._source_height_scale)
        font_form.addRow("Max height growth (× source height)", self._text_layout_max_height)
        font_form.addRow("Max width growth (× source width)", self._text_layout_widen)
        outer.addWidget(font_box)

        # OCR line grouping — the FIRST-layer merge that fuses adjacent
        # single-line OCR rows into a multi-line Observation. This is what
        # fuses menu items like CLEAR / SAVE / QUICKSAVE / … into one 10-line
        # block. Set both knobs to 0 to force every OCR row into its own
        # Observation (each menu item translated separately).
        ocr_merge_box = QtWidgets.QGroupBox("OCR line grouping (source-level)", self)
        ocr_merge_form = QtWidgets.QFormLayout(ocr_merge_box)
        self._ocr_line_min_gap = self._int_spin(0, 200, 1, int(getattr(config, "OCR_LINE_MERGE_MIN_GAP_PX", 4)))
        self._ocr_line_height_factor = QtWidgets.QDoubleSpinBox(self)
        self._ocr_line_height_factor.setDecimals(2)
        self._ocr_line_height_factor.setRange(0.0, 3.00)
        self._ocr_line_height_factor.setSingleStep(0.05)
        self._ocr_line_height_factor.setValue(float(getattr(config, "OCR_LINE_MERGE_LINE_HEIGHT_FACTOR", 0.35)))
        self._ocr_line_overlap_min = QtWidgets.QDoubleSpinBox(self)
        self._ocr_line_overlap_min.setDecimals(2)
        self._ocr_line_overlap_min.setRange(0.0, 1.00)
        self._ocr_line_overlap_min.setSingleStep(0.05)
        self._ocr_line_overlap_min.setValue(float(getattr(config, "OCR_LINE_MERGE_OVERLAP_MIN", 0.5)))
        ocr_merge_form.addRow("Min vertical gap allowance", self._ocr_line_min_gap)
        ocr_merge_form.addRow("Line-height multiplier", self._ocr_line_height_factor)
        ocr_merge_form.addRow("Min horizontal overlap", self._ocr_line_overlap_min)
        outer.addWidget(ocr_merge_box)

        # Row-merge tuning — the SECOND-layer merge that chains dialogue
        # tracks after Observations. Only affects dialogue-classified tracks;
        # menu rows go through the OCR-line-grouping stage above, not this
        # one.
        merge_box = QtWidgets.QGroupBox("Row merging (dialogue chain)", self)
        merge_form = QtWidgets.QFormLayout(merge_box)
        self._chain_gap_max = self._int_spin(0, 200, 1, int(getattr(config, "DIALOGUE_BOX_CHAIN_GAP_MAX_PX", 34)))
        self._chain_align_tol = self._int_spin(0, 400, 2, int(getattr(config, "DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX", 132)))
        merge_form.addRow("Max vertical gap for merge", self._chain_gap_max)
        merge_form.addRow("Horizontal align tolerance", self._chain_align_tol)
        outer.addWidget(merge_box)

        note = QtWidgets.QLabel(
            "Values apply on the next rendered frame and are saved when this window closes. "
            "Restart the app to reset to defaults by removing the value from config.runtime.json.",
            self,
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        outer.addWidget(buttons)

        for spin in (
            self._edge_padding,
            self._patch_expand_x,
            self._patch_expand_y,
            self._text_margin_x,
            self._text_margin_y,
            self._text_min_pixel,
            self._text_max_pixel,
            self._chain_gap_max,
            self._chain_align_tol,
            self._ocr_line_min_gap,
        ):
            spin.valueChanged.connect(self._apply_live)
        self._source_height_scale.valueChanged.connect(self._apply_live)
        self._ocr_line_height_factor.valueChanged.connect(self._apply_live)
        self._ocr_line_overlap_min.valueChanged.connect(self._apply_live)
        self._text_layout_max_height.valueChanged.connect(self._apply_live)
        self._text_layout_widen.valueChanged.connect(self._apply_live)

        self.setMinimumWidth(420)

    def _int_spin(self, lo: int, hi: int, step: int, value: int) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox(self)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(" px")
        return spin

    def _apply_live(self) -> None:
        # Push every knob into the config module so the next scene rebuild picks
        # them up. Persistence to disk is deferred to closeEvent so we don't
        # write config.runtime.json on every spin-box tick.
        config.EDGE_IGNORE_PADDING = int(self._edge_padding.value())
        config.PATCH_EXPAND_X = int(self._patch_expand_x.value())
        config.PATCH_EXPAND_Y = int(self._patch_expand_y.value())
        config.TEXT_MARGIN_X = int(self._text_margin_x.value())
        config.TEXT_MARGIN_Y = int(self._text_margin_y.value())
        min_px = int(self._text_min_pixel.value())
        max_px = int(self._text_max_pixel.value())
        if max_px < min_px:
            max_px = min_px
            self._text_max_pixel.blockSignals(True)
            self._text_max_pixel.setValue(max_px)
            self._text_max_pixel.blockSignals(False)
        config.TEXT_MIN_PIXEL = min_px
        config.TEXT_MAX_PIXEL = max_px
        config.TEXT_SOURCE_HEIGHT_SCALE = float(self._source_height_scale.value())
        config.TEXT_LAYOUT_MAX_HEIGHT_RATIO = float(self._text_layout_max_height.value())
        config.TEXT_LAYOUT_WIDEN_MAX_RATIO = float(self._text_layout_widen.value())
        config.DIALOGUE_BOX_CHAIN_GAP_MAX_PX = int(self._chain_gap_max.value())
        config.DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX = int(self._chain_align_tol.value())
        config.OCR_LINE_MERGE_MIN_GAP_PX = int(self._ocr_line_min_gap.value())
        config.OCR_LINE_MERGE_LINE_HEIGHT_FACTOR = float(self._ocr_line_height_factor.value())
        config.OCR_LINE_MERGE_OVERLAP_MIN = float(self._ocr_line_overlap_min.value())
        # Keep the existing edge-padding signal wire — controller listens on it
        # to recompute allowed regions.
        self.settingsApplied.emit(config.EDGE_IGNORE_PADDING)

    def _save(self) -> None:
        self._apply_live()
        config.save_runtime_settings(
            {
                "EDGE_IGNORE_PADDING": config.EDGE_IGNORE_PADDING,
                "PATCH_EXPAND_X": config.PATCH_EXPAND_X,
                "PATCH_EXPAND_Y": config.PATCH_EXPAND_Y,
                "TEXT_MARGIN_X": config.TEXT_MARGIN_X,
                "TEXT_MARGIN_Y": config.TEXT_MARGIN_Y,
                "TEXT_MIN_PIXEL": config.TEXT_MIN_PIXEL,
                "TEXT_MAX_PIXEL": config.TEXT_MAX_PIXEL,
                "TEXT_SOURCE_HEIGHT_SCALE": config.TEXT_SOURCE_HEIGHT_SCALE,
                "TEXT_LAYOUT_MAX_HEIGHT_RATIO": config.TEXT_LAYOUT_MAX_HEIGHT_RATIO,
                "TEXT_LAYOUT_WIDEN_MAX_RATIO": config.TEXT_LAYOUT_WIDEN_MAX_RATIO,
                "DIALOGUE_BOX_CHAIN_GAP_MAX_PX": config.DIALOGUE_BOX_CHAIN_GAP_MAX_PX,
                "DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX": config.DIALOGUE_BOX_CHAIN_ALIGN_TOLERANCE_PX,
                "OCR_LINE_MERGE_MIN_GAP_PX": config.OCR_LINE_MERGE_MIN_GAP_PX,
                "OCR_LINE_MERGE_LINE_HEIGHT_FACTOR": config.OCR_LINE_MERGE_LINE_HEIGHT_FACTOR,
                "OCR_LINE_MERGE_OVERLAP_MIN": config.OCR_LINE_MERGE_OVERLAP_MIN,
            }
        )
        self._saved = True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._save()
        super().closeEvent(event)


class ControlPanel(QtWidgets.QWidget):
    """Window picker, settings, and region controls for the single full-window workflow."""

    attachRequested = QtCore.pyqtSignal(int)
    pauseToggled = QtCore.pyqtSignal(bool)
    languageChanged = QtCore.pyqtSignal(str)
    settingsApplied = QtCore.pyqtSignal(int)
    zonesRequested = QtCore.pyqtSignal()

    def __init__(self, language_choices: list[tuple[str, str]] | None = None, current_lang: str = "ja"):
        super().__init__()
        self.setWindowTitle(f"OverlayTranslator2 v{VERSION} - Controls")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._paused = False
        self._attached = False
        self._language_choices = language_choices or [
            ("ja", "Japanese"),
            ("ko", "Korean"),
            ("zh-Hans", "Chinese (Simplified)"),
            ("zh-Hant", "Chinese (Traditional)"),
            ("en", "English"),
            ("0", "Auto detect / no hint"),
        ]
        self._settings_dialog: SettingsDialog | None = None

        frame = QtWidgets.QFrame(self)
        frame.setObjectName("frame")
        frame.setGeometry(0, 0, 520, 340)
        frame.setStyleSheet(
            "QFrame#frame { background: rgba(20,20,20,210); border: 1px solid rgba(255,255,255,50); border-radius: 12px; }"
            "QPushButton, QComboBox, QSpinBox { background: rgba(255,255,255,22); color: white; border: 1px solid rgba(255,255,255,40); border-radius: 6px; padding: 5px; }"
            "QPushButton:disabled { color: rgba(255,255,255,90); background: rgba(255,255,255,10); }"
            "QLabel { color: white; }"
        )

        self._move = MoveHandle(self, tooltip="Move controls")
        self._move.move(10, 10)

        self._title = QtWidgets.QLabel("Target window", self)
        self._title.setGeometry(40, 10, 160, 22)

        self._combo = WindowPickerCombo(self)
        self._combo.setGeometry(14, 42, 492, 30)

        self._lang_label = QtWidgets.QLabel("Source hint", self)
        self._lang_label.setGeometry(14, 78, 120, 20)

        self._lang_combo = QtWidgets.QComboBox(self)
        self._lang_combo.setGeometry(14, 100, 492, 30)
        for tag, label in self._language_choices:
            self._lang_combo.addItem(label, tag)
        current_idx = 0
        for idx in range(self._lang_combo.count()):
            if str(self._lang_combo.itemData(idx)) == str(current_lang or "0"):
                current_idx = idx
                break
        self._lang_combo.setCurrentIndex(current_idx)
        self._lang_combo.currentIndexChanged.connect(self._emit_language_change)

        self._attach = QtWidgets.QPushButton("Attach", self)
        self._attach.setGeometry(14, 140, 240, 32)
        self._attach.clicked.connect(self.attach_selected)

        self._pause = QtWidgets.QPushButton("Pause", self)
        self._pause.setGeometry(262, 140, 122, 32)
        self._pause.clicked.connect(self._toggle_pause)
        self._pause.setEnabled(False)

        self._settings = QtWidgets.QPushButton("Settings", self)
        self._settings.setGeometry(392, 140, 114, 32)
        self._settings.clicked.connect(self._open_settings)

        self._zones = QtWidgets.QPushButton("Zones", self)
        self._zones.setGeometry(14, 184, 492, 34)
        self._zones.clicked.connect(self.zonesRequested.emit)

        self._hint = QtWidgets.QLabel(
            "Open Zones to manage multiple ignore and translation zones. "
            "Configured zones stay invisible during normal overlay and appear only in debug images.",
            self,
        )
        self._hint.setWordWrap(True)
        self._hint.setGeometry(14, 226, 492, 42)

        self._status = QtWidgets.QLabel("Choose a target window.", self)
        self._status.setWordWrap(True)
        self._status.setGeometry(14, 274, 492, 52)

        self.setFixedSize(520, 340)
        self._zones.setEnabled(False)
        self._combo.opened.connect(lambda: self.refresh_windows(preserve_current=True))
        self.refresh_windows()

    def refresh_windows(self, preserve_current: bool = False):
        previous_hwnd = int(self._combo.currentData() or 0) if preserve_current else 0
        self._combo.clear()
        windows = list_candidate_windows()
        selected_index = -1
        for idx, win in enumerate(windows):
            label = f"{win.title}  [PID {win.pid}]"
            self._combo.addItem(label, win.hwnd)
            if previous_hwnd and int(win.hwnd) == previous_hwnd:
                selected_index = idx
        if selected_index >= 0:
            self._combo.setCurrentIndex(selected_index)
        self._status.setText(
            f"Found {len(windows)} visible windows. Attach to the app you want to translate. Opening the list auto-refreshes it."
        )

    def _emit_language_change(self):
        tag = str(self._lang_combo.currentData() or "0")
        self.languageChanged.emit(tag)

    def attach_selected(self):
        hwnd = int(self._combo.currentData() or 0)
        if not hwnd:
            self._status.setText("Choose a target window first.")
            return
        self.attachRequested.emit(hwnd)

    def _toggle_pause(self):
        self._paused = not self._paused
        self._pause.setText("Resume" if self._paused else "Pause")
        self.pauseToggled.emit(self._paused)

    def _open_settings(self):
        if self._settings_dialog is None:
            self._settings_dialog = SettingsDialog(self)
            self._settings_dialog.settingsApplied.connect(self.settingsApplied.emit)
        self._settings_dialog.show()
        self._settings_dialog.raise_()
        self._settings_dialog.activateWindow()

    def set_status(self, text: str):
        self._status.setText(text)

    def set_attached(self, attached: bool):
        self._attached = bool(attached)
        self._paused = False
        self._pause.setText("Pause")
        self._pause.setEnabled(bool(attached))
        self._zones.setEnabled(bool(attached))

    def set_paused_state(self, paused: bool):
        self._paused = bool(paused)
        self._pause.setText("Resume" if self._paused else "Pause")

    @QtCore.pyqtSlot()
    def toggle_pause(self):
        self._toggle_pause()


class RegionSelectionWindow(QtWidgets.QWidget):
    """Dedicated region editor overlay that captures mouse input reliably."""

    regionEdited = QtCore.pyqtSignal(str, int, int, int, int)
    regionEditCancelled = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"OverlayTranslator2 v{VERSION} - Region Editor")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._target_hwnd = 0
        self._ignore_regions: list[QtCore.QRect] = []
        self._translation_zones: list[dict[str, object]] = []
        self._edit_kind: str | None = None
        self._draft_region = QtCore.QRect()
        self._drag_mode: str | None = None
        self._drag_handle: str | None = None
        self._drag_anchor = QtCore.QPoint()
        self._drag_start_rect = QtCore.QRect()
        self._edit_original_rect = QtCore.QRect()
        self.hide()

    def set_target_window(self, hwnd: int) -> None:
        self._target_hwnd = int(hwnd or 0)
        self._sync_z_order()

    def set_client_region(self, left: int, top: int, width: int, height: int) -> None:
        self.setGeometry(int(left), int(top), int(width), int(height))
        if self._edit_kind and width > 0 and height > 0:
            self.show()
            self.raise_()
            self.activateWindow()
        self.update()

    def set_ignore_regions(self, regions: list[tuple[int, int, int, int]] | list[dict[str, int]]) -> None:
        normalized: list[QtCore.QRect] = []
        for item in regions or []:
            if isinstance(item, dict):
                rect = QtCore.QRect(
                    int(item.get("left", 0)),
                    int(item.get("top", 0)),
                    int(item.get("width", 0)),
                    int(item.get("height", 0)),
                ).normalized()
            else:
                left, top, width, height = item
                rect = QtCore.QRect(int(left), int(top), int(width), int(height)).normalized()
            if rect.width() >= 1 and rect.height() >= 1:
                normalized.append(rect)
        self._ignore_regions = normalized
        self.update()

    def set_translation_zones(self, zones: list[dict[str, object]]) -> None:
        normalized: list[dict[str, object]] = []
        for item in zones or []:
            rect = QtCore.QRect(
                _d_int(item.get("left", 0)),
                _d_int(item.get("top", 0)),
                _d_int(item.get("width", 0)),
                _d_int(item.get("height", 0)),
            ).normalized()
            if rect.width() >= 1 and rect.height() >= 1:
                normalized.append({"rect": rect, "group_all": bool(item.get("group_all", False))})
        self._translation_zones = normalized
        self.update()

    def begin_region_edit(self, kind: str, rect: tuple[int, int, int, int] | None = None) -> None:
        if kind not in {"ignore", "limit"}:
            return
        self._edit_kind = kind
        if rect is not None:
            self._draft_region = QtCore.QRect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])).normalized()
            self._edit_original_rect = QtCore.QRect(self._draft_region)
        else:
            self._draft_region = QtCore.QRect()
            self._edit_original_rect = QtCore.QRect()
        self._drag_mode = None
        self._drag_handle = None
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
        self.grabKeyboard()
        self._sync_z_order()
        self.update()

    def lock_region_edit(self, kind: str) -> None:
        if self._edit_kind != kind:
            return
        region = self._snap_avoid_overlap(self._draft_region.normalized())
        if region.width() < 8 or region.height() < 8:
            self.cancel_region_edit(kind)
            return
        self._edit_kind = None
        self._drag_mode = None
        self._drag_handle = None
        self.unsetCursor()
        if self.keyboardGrabber() is self:
            self.releaseKeyboard()
        self.hide()
        self.regionEdited.emit(kind, region.left(), region.top(), region.width(), region.height())

    def cancel_region_edit(self, kind: str | None = None) -> None:
        kind = kind or self._edit_kind
        if not kind:
            return
        self._edit_kind = None
        self._drag_mode = None
        self._drag_handle = None
        self._draft_region = QtCore.QRect()
        self._edit_original_rect = QtCore.QRect()
        self.unsetCursor()
        if self.keyboardGrabber() is self:
            self.releaseKeyboard()
        self.hide()
        self.regionEditCancelled.emit(kind)

    def _sync_z_order(self) -> None:
        if not self.isVisible() or not self._target_hwnd:
            return
        try:
            stack_window_above_target(int(self.winId()), int(self._target_hwnd))
            self.raise_()
        except Exception:
            pass

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        try:
            apply_exclude_from_capture(int(self.winId()))
        except Exception:
            pass
        self._sync_z_order()

    def _handle_rects(self, rect: QtCore.QRect) -> dict[str, QtCore.QRect]:
        if rect.isNull():
            return {}
        size = 12
        half = size // 2
        points = {
            "nw": rect.topLeft(),
            "ne": rect.topRight(),
            "sw": rect.bottomLeft(),
            "se": rect.bottomRight(),
            "n": QtCore.QPoint(rect.center().x(), rect.top()),
            "s": QtCore.QPoint(rect.center().x(), rect.bottom()),
            "w": QtCore.QPoint(rect.left(), rect.center().y()),
            "e": QtCore.QPoint(rect.right(), rect.center().y()),
        }
        return {name: QtCore.QRect(point.x() - half, point.y() - half, size, size) for name, point in points.items()}

    def _hit_handle(self, pos: QtCore.QPoint) -> str | None:
        draft = self._draft_region.normalized()
        for name, handle in self._handle_rects(draft).items():
            if handle.contains(pos):
                return name
        return None

    def _clamp_point(self, point: QtCore.QPoint) -> QtCore.QPoint:
        return QtCore.QPoint(max(0, min(self.width() - 1, point.x())), max(0, min(self.height() - 1, point.y())))

    def _occupied_rects(self) -> list[QtCore.QRect]:
        occupied: list[QtCore.QRect] = []
        for zone in self._translation_zones:
            zone_rect = zone["rect"]
            assert isinstance(zone_rect, QtCore.QRect), "zone['rect'] must be a QRect at runtime"
            rect = QtCore.QRect(zone_rect).normalized()
            if not self._edit_original_rect.isNull() and rect == self._edit_original_rect:
                continue
            occupied.append(rect)
        for rect in self._ignore_regions:
            rect = QtCore.QRect(rect).normalized()
            if not self._edit_original_rect.isNull() and rect == self._edit_original_rect:
                continue
            occupied.append(rect)
        return occupied

    def _clamp_rect(self, rect: QtCore.QRect) -> QtCore.QRect:
        rect = rect.normalized()
        if rect.width() < 1 or rect.height() < 1:
            return rect
        if rect.left() < 0:
            rect.moveLeft(0)
        if rect.top() < 0:
            rect.moveTop(0)
        if rect.right() >= self.width():
            rect.moveRight(max(0, self.width() - 1))
        if rect.bottom() >= self.height():
            rect.moveBottom(max(0, self.height() - 1))
        return rect

    def _rect_overlaps_any(self, rect: QtCore.QRect, occupied: list[QtCore.QRect], *, margin: int = 0) -> bool:
        rect = rect.normalized()
        for other in occupied:
            a_left = rect.left() - margin
            a_top = rect.top() - margin
            a_right = rect.right() + margin
            a_bottom = rect.bottom() + margin
            b_left = other.left() - margin
            b_top = other.top() - margin
            b_right = other.right() + margin
            b_bottom = other.bottom() + margin
            if a_left <= b_right and a_right >= b_left and a_top <= b_bottom and a_bottom >= b_top:
                return True
        return False

    def _snap_move_no_overlap(self, rect: QtCore.QRect, occupied: list[QtCore.QRect]) -> QtCore.QRect:
        current = self._clamp_rect(rect.normalized())
        if not occupied or not self._rect_overlaps_any(current, occupied):
            return current
        original = QtCore.QRect(current)
        for _ in range(32):
            overlaps = [other for other in occupied if self._rect_overlaps_any(current, [other])]
            if not overlaps:
                break
            candidates: list[QtCore.QRect] = []
            for other in overlaps:
                candidates.extend(
                    [
                        self._clamp_rect(
                            QtCore.QRect(
                                other.left() - current.width(), current.top(), current.width(), current.height()
                            )
                        ),
                        self._clamp_rect(
                            QtCore.QRect(other.right() + 1, current.top(), current.width(), current.height())
                        ),
                        self._clamp_rect(
                            QtCore.QRect(
                                current.left(), other.top() - current.height(), current.width(), current.height()
                            )
                        ),
                        self._clamp_rect(
                            QtCore.QRect(current.left(), other.bottom() + 1, current.width(), current.height())
                        ),
                    ]
                )
            valid = [
                c for c in candidates if c.width() >= 1 and c.height() >= 1 and not self._rect_overlaps_any(c, occupied)
            ]
            if not valid:
                break
            current = min(valid, key=lambda r: abs(r.left() - original.left()) + abs(r.top() - original.top()))
        return current

    def _snap_resize_no_overlap(
        self, rect: QtCore.QRect, occupied: list[QtCore.QRect], handle: str | None
    ) -> QtCore.QRect:
        current = self._clamp_rect(rect.normalized())
        if not occupied or not self._rect_overlaps_any(current, occupied):
            return current
        handle = str(handle or "")
        for other in occupied:
            if not self._rect_overlaps_any(current, [other]):
                continue
            horizontal_overlap = not (current.right() < other.left() or current.left() > other.right())
            vertical_overlap = not (current.bottom() < other.top() or current.top() > other.bottom())
            if "e" in handle and vertical_overlap and current.left() < other.left():
                current.setRight(max(current.left(), other.left() - 1))
            if "w" in handle and vertical_overlap and current.right() > other.right():
                current.setLeft(min(current.right(), other.right() + 1))
            if "s" in handle and horizontal_overlap and current.top() < other.top():
                current.setBottom(max(current.top(), other.top() - 1))
            if "n" in handle and horizontal_overlap and current.bottom() > other.bottom():
                current.setTop(min(current.bottom(), other.bottom() + 1))
            current = self._clamp_rect(current.normalized())
        if self._rect_overlaps_any(current, occupied):
            return self._snap_move_no_overlap(current, occupied)
        return current

    def _snap_avoid_overlap(
        self, rect: QtCore.QRect, *, mode: str | None = None, handle: str | None = None
    ) -> QtCore.QRect:
        rect = self._clamp_rect(rect.normalized())
        occupied = self._occupied_rects()
        if not occupied:
            return rect
        if not self._rect_overlaps_any(rect, occupied):
            return rect
        if mode == "resize":
            snapped = self._snap_resize_no_overlap(rect, occupied, handle)
        else:
            snapped = self._snap_move_no_overlap(rect, occupied)
        return self._clamp_rect(snapped.normalized())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._edit_kind is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            event.ignore()
            return
        event.accept()
        pos = self._clamp_point(event.position().toPoint())
        self._drag_anchor = pos
        self._drag_start_rect = QtCore.QRect(self._draft_region)
        handle = self._hit_handle(pos)
        if handle:
            self._drag_mode = "resize"
            self._drag_handle = handle
        elif not self._draft_region.isNull() and self._draft_region.normalized().contains(pos):
            self._drag_mode = "move"
            self._drag_handle = None
        else:
            self._drag_mode = "new"
            self._drag_handle = None
            self._draft_region = QtCore.QRect(pos, pos)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._edit_kind is None or not self._drag_mode:
            event.ignore()
            return
        event.accept()
        pos = self._clamp_point(event.position().toPoint())
        if self._drag_mode == "new":
            self._draft_region = self._snap_avoid_overlap(QtCore.QRect(self._drag_anchor, pos).normalized(), mode="new")
        elif self._drag_mode == "move":
            delta = pos - self._drag_anchor
            rect = QtCore.QRect(self._drag_start_rect)
            rect.translate(delta)
            self._draft_region = self._snap_avoid_overlap(self._clamp_rect(rect), mode="move")
        elif self._drag_mode == "resize":
            rect = QtCore.QRect(self._drag_start_rect)
            handle = self._drag_handle or ""
            if "n" in handle:
                rect.setTop(pos.y())
            if "s" in handle:
                rect.setBottom(pos.y())
            if "w" in handle:
                rect.setLeft(pos.x())
            if "e" in handle:
                rect.setRight(pos.x())
            self._draft_region = self._snap_avoid_overlap(rect.normalized(), mode="resize", handle=handle)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._edit_kind is None:
            event.ignore()
            return
        event.accept()
        self._drag_mode = None
        self._drag_handle = None
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == QtCore.Qt.Key.Key_Escape and self._edit_kind:
            kind = self._edit_kind
            self.cancel_region_edit(kind)
            event.accept()
            return
        super().keyPressEvent(event)

    def _draw_region(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRect,
        *,
        color: QtGui.QColor,
        label: str,
        fill_alpha: int,
        editing: bool = False,
    ) -> None:
        if rect.isNull():
            return
        painter.save()
        fill = QtGui.QColor(color)
        fill.setAlpha(fill_alpha)
        painter.fillRect(rect, fill)
        pen = QtGui.QPen(color, 2 if editing else 1)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.drawText(
            rect.adjusted(6, 4, -6, -4), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, label
        )
        if editing:
            for handle in self._handle_rects(rect).values():
                painter.fillRect(handle, color)
        painter.restore()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        if not self._edit_kind:
            return
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(8, 12, 18, 72))
        for idx, zone in enumerate(self._translation_zones, start=1):
            rect = zone["rect"]
            assert isinstance(rect, QtCore.QRect), "zone['rect'] must be a QRect"
            label = f"Translation {idx}" + (" [group]" if bool(zone.get("group_all", False)) else "")
            self._draw_region(painter, rect, color=QtGui.QColor(64, 200, 255), label=label, fill_alpha=18)
        for idx, rect in enumerate(self._ignore_regions, start=1):
            self._draw_region(painter, rect, color=QtGui.QColor(255, 96, 96), label=f"Ignore {idx}", fill_alpha=18)
        hint = "Drag inside the hooked window to draw. Drag inside the box to move it. Drag handles to resize. Click Lock Region in Zones to apply. Press Esc to cancel."
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(
            self.rect().adjusted(12, 12, -12, -12),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            hint,
        )
        draft = self._draft_region.normalized()
        if not draft.isNull():
            self._draw_region(
                painter,
                draft,
                color=(QtGui.QColor(64, 200, 255) if self._edit_kind == "limit" else QtGui.QColor(255, 96, 96)),
                label=("Translation zone" if self._edit_kind == "limit" else "Ignore zone"),
                fill_alpha=40,
                editing=True,
            )


class ZoneRowWidget(QtWidgets.QFrame):
    editRequested = QtCore.pyqtSignal(int)
    removeRequested = QtCore.pyqtSignal(int)
    groupToggled = QtCore.pyqtSignal(int, bool)

    def __init__(
        self, kind: str, index: int, label: str, *, grouped: bool = False, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent)
        self._kind = str(kind)
        self._index = int(index)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: rgba(255,255,255,10); border: 1px solid rgba(255,255,255,30); border-radius: 6px; }"
            "QLabel { color: white; }"
            "QPushButton, QCheckBox { color: white; }"
        )
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        self._label = QtWidgets.QLabel(label, self)
        self._label.setWordWrap(True)
        layout.addWidget(self._label, 1)
        self._group = None
        if self._kind == "limit":
            self._group = QtWidgets.QCheckBox("Group", self)
            self._group.setChecked(bool(grouped))
            self._group.toggled.connect(lambda checked: self.groupToggled.emit(self._index, bool(checked)))
            layout.addWidget(self._group)
        self._edit = QtWidgets.QPushButton("Edit", self)
        self._remove = QtWidgets.QPushButton("Remove", self)
        self._edit.clicked.connect(lambda: self.editRequested.emit(self._index))
        self._remove.clicked.connect(lambda: self.removeRequested.emit(self._index))
        layout.addWidget(self._edit)
        layout.addWidget(self._remove)

    def setEditingEnabled(self, enabled: bool) -> None:
        if self._group is not None:
            self._group.setEnabled(bool(enabled))
        self._edit.setEnabled(bool(enabled))
        self._remove.setEnabled(bool(enabled))


class ZoneManagerWindow(QtWidgets.QWidget):
    zoneListsChanged = QtCore.pyqtSignal(object, object)
    beginZoneEditRequested = QtCore.pyqtSignal(str, object, object)
    lockZoneEditRequested = QtCore.pyqtSignal(str)
    cancelZoneEditRequested = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"OverlayTranslator2 v{VERSION} - Zones")
        self.setWindowFlags(QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self._ignore_regions: list[tuple[int, int, int, int]] = []
        self._translation_zones: list[dict[str, object]] = []
        self._editing_kind: str | None = None
        self._editing_index: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(
            "Use this window to manage multiple ignore and translation zones. "
            "Zones stay invisible during normal overlay and are shown in debug images.",
            self,
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        split = QtWidgets.QHBoxLayout()
        layout.addLayout(split)

        ignore_box = QtWidgets.QGroupBox("Ignore zones", self)
        ignore_layout = QtWidgets.QVBoxLayout(ignore_box)
        self._ignore_add = QtWidgets.QPushButton("Add Ignore Zone", self)
        self._ignore_add.clicked.connect(self._on_ignore_button)
        ignore_layout.addWidget(self._ignore_add)
        self._ignore_scroll = QtWidgets.QScrollArea(self)
        self._ignore_scroll.setWidgetResizable(True)
        self._ignore_container = QtWidgets.QWidget(self)
        self._ignore_rows = QtWidgets.QVBoxLayout(self._ignore_container)
        self._ignore_rows.setContentsMargins(0, 0, 0, 0)
        self._ignore_rows.setSpacing(8)
        self._ignore_rows.addStretch(1)
        self._ignore_scroll.setWidget(self._ignore_container)
        ignore_layout.addWidget(self._ignore_scroll, 1)
        split.addWidget(ignore_box)

        trans_box = QtWidgets.QGroupBox("Translation zones", self)
        trans_layout = QtWidgets.QVBoxLayout(trans_box)
        self._translation_add = QtWidgets.QPushButton("Add Translation Zone", self)
        self._translation_add.clicked.connect(self._on_translation_button)
        trans_layout.addWidget(self._translation_add)
        self._translation_scroll = QtWidgets.QScrollArea(self)
        self._translation_scroll.setWidgetResizable(True)
        self._translation_container = QtWidgets.QWidget(self)
        self._translation_rows = QtWidgets.QVBoxLayout(self._translation_container)
        self._translation_rows.setContentsMargins(0, 0, 0, 0)
        self._translation_rows.setSpacing(8)
        self._translation_rows.addStretch(1)
        self._translation_scroll.setWidget(self._translation_container)
        trans_layout.addWidget(self._translation_scroll, 1)
        split.addWidget(trans_box)

        footer = QtWidgets.QHBoxLayout()
        self._lock = QtWidgets.QPushButton("Lock Region", self)
        self._cancel = QtWidgets.QPushButton("Cancel Edit", self)
        self._close = QtWidgets.QPushButton("Close", self)
        footer.addWidget(self._lock)
        footer.addWidget(self._cancel)
        footer.addStretch(1)
        footer.addWidget(self._close)
        layout.addLayout(footer)

        self._lock.clicked.connect(lambda: self.lockZoneEditRequested.emit(self._editing_kind or ""))
        self._cancel.clicked.connect(lambda: self.cancelZoneEditRequested.emit(self._editing_kind or ""))
        self._close.clicked.connect(self.hide)
        self._lock.hide()
        self.resize(840, 520)
        self._refresh_ui()

    def _on_ignore_button(self) -> None:
        if self._editing_kind == "ignore":
            self.lockZoneEditRequested.emit("ignore")
            return
        if self._editing_kind is None:
            self._start_edit("ignore", None)

    def _on_translation_button(self) -> None:
        if self._editing_kind == "limit":
            self.lockZoneEditRequested.emit("limit")
            return
        if self._editing_kind is None:
            self._start_edit("limit", None)

    def _zone_label(self, index: int, rect: tuple[int, int, int, int], *, grouped: bool = False) -> str:
        left, top, width, height = rect
        suffix = "  [group]" if grouped else ""
        return f"{index + 1:02d}  {left}, {top}, {width}, {height}{suffix}"

    def set_ignore_regions(self, regions: list[tuple[int, int, int, int]] | list[dict[str, int]]) -> None:
        normalized: list[tuple[int, int, int, int]] = []
        for item in regions or []:
            if isinstance(item, dict):
                normalized.append(
                    (
                        int(item.get("left", 0)),
                        int(item.get("top", 0)),
                        int(item.get("width", 0)),
                        int(item.get("height", 0)),
                    )
                )
            else:
                vals = [int(v) for v in item]
                if len(vals) >= 4:
                    normalized.append((vals[0], vals[1], vals[2], vals[3]))
        self._ignore_regions = normalized
        self._rebuild_lists()

    def set_translation_zones(self, zones: list[dict[str, object]]) -> None:
        normalized: list[dict[str, object]] = []
        for item in zones or []:
            normalized.append(
                {
                    "left": _d_int(item.get("left", 0)),
                    "top": _d_int(item.get("top", 0)),
                    "width": _d_int(item.get("width", 0)),
                    "height": _d_int(item.get("height", 0)),
                    "group_all": bool(item.get("group_all", False)),
                }
            )
        self._translation_zones = normalized
        self._rebuild_lists()

    def ignore_regions(self) -> list[tuple[int, int, int, int]]:
        return list(self._ignore_regions)

    def translation_zones(self) -> list[dict[str, object]]:
        return [dict(item) for item in self._translation_zones]

    def _clear_layout_rows(self, layout: QtWidgets.QVBoxLayout) -> None:
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_lists(self) -> None:
        self._clear_layout_rows(self._ignore_rows)
        for idx, rect in enumerate(self._ignore_regions):
            row = ZoneRowWidget("ignore", idx, self._zone_label(idx, rect), parent=self._ignore_container)
            row.editRequested.connect(lambda index, kind="ignore": self._start_edit(kind, index))
            row.removeRequested.connect(self._remove_ignore_zone)
            row.setEditingEnabled(self._editing_kind is None)
            self._ignore_rows.insertWidget(max(0, self._ignore_rows.count() - 1), row)
        self._clear_layout_rows(self._translation_rows)
        for idx, zone in enumerate(self._translation_zones):
            rect = (_d_int(zone["left"]), _d_int(zone["top"]), _d_int(zone["width"]), _d_int(zone["height"]))
            row = ZoneRowWidget(
                "limit",
                idx,
                self._zone_label(idx, rect, grouped=bool(zone.get("group_all", False))),
                grouped=bool(zone.get("group_all", False)),
                parent=self._translation_container,
            )
            row.editRequested.connect(lambda index, kind="limit": self._start_edit(kind, index))
            row.removeRequested.connect(self._remove_translation_zone)
            row.groupToggled.connect(self._on_group_toggled)
            row.setEditingEnabled(self._editing_kind is None)
            self._translation_rows.insertWidget(max(0, self._translation_rows.count() - 1), row)
        self._refresh_ui()

    def _emit_zone_lists_changed(self) -> None:
        self.zoneListsChanged.emit(self.ignore_regions(), self.translation_zones())

    def _start_edit(self, kind: str, index: int | None) -> None:
        if self._editing_kind is not None:
            return
        if index is not None and index < 0:
            return
        rect = None
        if kind == "ignore" and index is not None and 0 <= index < len(self._ignore_regions):
            rect = self._ignore_regions[index]
        elif kind == "limit" and index is not None and 0 <= index < len(self._translation_zones):
            zone = self._translation_zones[index]
            rect = (_d_int(zone["left"]), _d_int(zone["top"]), _d_int(zone["width"]), _d_int(zone["height"]))
        self._editing_kind = kind
        self._editing_index = index
        self._refresh_ui()
        self.beginZoneEditRequested.emit(kind, index, rect)

    def apply_region_edit(self, kind: str, left: int, top: int, width: int, height: int) -> None:
        rect = (int(left), int(top), int(width), int(height))
        index = self._editing_index
        if kind == "ignore":
            if index is None or index < 0 or index >= len(self._ignore_regions):
                self._ignore_regions.append(rect)
            else:
                self._ignore_regions[index] = rect
        else:
            grouped = False
            if index is not None and 0 <= index < len(self._translation_zones):
                grouped = bool(self._translation_zones[index].get("group_all", False))
            zone: dict[str, object] = {
                "left": rect[0],
                "top": rect[1],
                "width": rect[2],
                "height": rect[3],
                "group_all": grouped,
            }
            if index is None or index < 0 or index >= len(self._translation_zones):
                self._translation_zones.append(zone)
            else:
                self._translation_zones[index] = zone
        self.finish_region_edit(cancelled=False)
        self._emit_zone_lists_changed()

    def finish_region_edit(self, cancelled: bool = False) -> None:
        self._editing_kind = None
        self._editing_index = None
        self._refresh_ui()
        self._rebuild_lists()

    def _remove_ignore_zone(self, row: int) -> None:
        if row < 0 or row >= len(self._ignore_regions):
            return
        self._ignore_regions.pop(row)
        self._rebuild_lists()
        self._emit_zone_lists_changed()

    def _remove_translation_zone(self, row: int) -> None:
        if row < 0 or row >= len(self._translation_zones):
            return
        self._translation_zones.pop(row)
        self._rebuild_lists()
        self._emit_zone_lists_changed()

    def _on_group_toggled(self, row: int, checked: bool) -> None:
        if self._editing_kind is not None:
            return
        if row < 0 or row >= len(self._translation_zones):
            return
        self._translation_zones[row]["group_all"] = bool(checked)
        self._rebuild_lists()
        self._emit_zone_lists_changed()

    def _refresh_ui(self) -> None:
        editing = self._editing_kind is not None
        editing_ignore = self._editing_kind == "ignore"
        editing_limit = self._editing_kind == "limit"
        self._ignore_add.setText("Lock Region" if editing_ignore else "Add Ignore Zone")
        self._translation_add.setText("Lock Region" if editing_limit else "Add Translation Zone")
        self._ignore_add.setEnabled((not editing) or editing_ignore)
        self._translation_add.setEnabled((not editing) or editing_limit)
        self._lock.setVisible(False)
        self._lock.setEnabled(False)
        self._cancel.setEnabled(editing)
        for layout in (self._ignore_rows, self._translation_rows):
            for i in range(max(0, layout.count() - 1)):
                item = layout.itemAt(i)
                if item is None:
                    continue
                widget = item.widget()
                if isinstance(widget, ZoneRowWidget):
                    widget.setEditingEnabled(not editing)


class TranslationOverlay(QtWidgets.QWidget):
    """Transparent full-window overlay that redraws translated text in place."""

    def __init__(self):
        super().__init__()
        flags = QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Tool
        transparent_input_flag = getattr(QtCore.Qt.WindowType, "WindowTransparentForInput", None)
        if transparent_input_flag is not None:
            flags |= transparent_input_flag
        self.setWindowFlags(flags)
        self.setWindowTitle(f"OverlayTranslator2 v{VERSION} - Translation")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NativeWindow, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        show_without_activating = getattr(QtCore.Qt.WidgetAttribute, "WA_ShowWithoutActivating", None)
        if show_without_activating is not None:
            self.setAttribute(show_without_activating, True)
        self._items: list[dict] = []
        self._target_hwnd = 0
        self._owner_hwnd = 0
        self._is_topmost = False
        self._allowed_regions: list[QtCore.QRect] = []
        self._ignore_regions: list[QtCore.QRect] = []
        # Poll foreground state to catch focus changes that don't coincide with
        # a scene update — otherwise the overlay would stay topmost until the
        # next OCR tick and briefly cover the app the user Alt-Tabbed to.
        self._zorder_timer = QtCore.QTimer(self)
        self._zorder_timer.setInterval(350)
        self._zorder_timer.timeout.connect(self._sync_z_order)
        self.hide()

    def _has_overlay_content(self) -> bool:
        return bool(self._items)

    def _refresh_visibility(self) -> None:
        if self._has_overlay_content() and self.width() > 0 and self.height() > 0:
            self.show()
            self._sync_z_order()
            if self._target_hwnd and not self._zorder_timer.isActive():
                self._zorder_timer.start()
        else:
            self._zorder_timer.stop()
            if self._is_topmost:
                try:
                    set_window_topmost(int(self.winId()), False)
                except Exception:
                    pass
                self._is_topmost = False
            self.hide()
        self.update()

    def _sync_owner(self):
        overlay_hwnd = int(self.winId()) if int(self.winId()) else 0
        target_hwnd = int(self._target_hwnd or 0)
        if not overlay_hwnd:
            return
        if target_hwnd == int(self._owner_hwnd or 0):
            return
        try:
            if set_owner_window(overlay_hwnd, target_hwnd):
                self._owner_hwnd = target_hwnd
        except Exception:
            pass

    def set_target_window(self, hwnd: int):
        self._target_hwnd = int(hwnd or 0)
        self._sync_owner()
        self._sync_z_order()

    def _sync_z_order(self):
        self._sync_owner()
        if not self.isVisible() or not self._target_hwnd:
            return
        try:
            overlay_hwnd = int(self.winId())
            want_topmost = is_target_foreground(int(self._target_hwnd))
            # Topmost while target has focus so no unrelated program can slip
            # over the translated text; drop back to the target's normal band
            # otherwise so we don't hover over whatever the user Alt-Tabbed to.
            if want_topmost != self._is_topmost:
                set_window_topmost(overlay_hwnd, want_topmost)
                self._is_topmost = want_topmost
            if not want_topmost:
                stack_window_above_target(overlay_hwnd, int(self._target_hwnd))
            self.raise_()
        except Exception:
            pass

    def showEvent(self, e: QtGui.QShowEvent):  # type: ignore[override]
        super().showEvent(e)
        self._sync_owner()
        try:
            apply_exclude_from_capture(int(self.winId()))
        except Exception:
            pass
        self._sync_z_order()

    def set_client_region(self, left: int, top: int, width: int, height: int):
        self.setGeometry(left, top, width, height)
        self._refresh_visibility()

    def set_scene(self, scene):
        items: list[dict] = []
        for item in getattr(scene, "items", []):
            extra_pairs = [
                (rect, pil_to_qimage(img)) for rect, img in getattr(item, "extra_patches", []) or []
            ]
            items.append(
                {
                    "patch_rect": item.patch_rect,
                    "patch_image": pil_to_qimage(item.patch_image),
                    "extra_patches": extra_pairs,
                    "text_rect": item.text_rect,
                    "text": item.text,
                    "fill": item.fill_color,
                    "outline": item.outline_color,
                    "allow_wrap": getattr(item, "allow_wrap", True),
                    "preferred_pixel_size": getattr(item, "preferred_pixel_size", 0),
                    "alignment": getattr(item, "alignment", "center"),
                    "font_family": getattr(item, "font_family", ""),
                    "font_weight": getattr(item, "font_weight", 0),
                    "writing_mode": getattr(item, "writing_mode", "horizontal"),
                }
            )
        self._items = items
        self._allowed_regions = [
            QtCore.QRect(r.left, r.top, r.width, r.height) for r in getattr(scene, "allowed_regions", [])
        ]
        self._ignore_regions = [
            QtCore.QRect(r.left, r.top, r.width, r.height) for r in getattr(scene, "ignore_regions", [])
        ]
        self._refresh_visibility()

    def _make_clip_path(self) -> QtGui.QPainterPath | None:
        if not self._allowed_regions:
            return None
        path = QtGui.QPainterPath()
        for rect in self._allowed_regions:
            path.addRect(QtCore.QRectF(rect))
        for rect in self._ignore_regions:
            ignore_path = QtGui.QPainterPath()
            ignore_path.addRect(QtCore.QRectF(rect))
            path = path.subtracted(ignore_path)
        return path

    def paintEvent(self, e: QtGui.QPaintEvent):  # type: ignore[override]
        painter = QtGui.QPainter(self)
        clip_path = self._make_clip_path()
        if clip_path is not None:
            painter.setClipPath(clip_path)
        if self._items:
            paint_overlay_items(painter, self._items)
