"""OneOcr Python package.

This package mirrors the PowerShell library split:
- common.py        <-> OneOcr.Common.psm1
- runtime_cache.py <-> OneOcr.RuntimeCache.psm1
- image.py         <-> OneOcr.Image.psm1
- native.py        <-> OneOcr.Native.psm1
- ui.py            <-> OneOcr.Ui.psm1

Design note:
This file intentionally avoids importing `native.py` at package import time.
That keeps startup lightweight and prevents Pillow from becoming a hard import
requirement just to import unrelated helpers such as `oneocr.common`.
"""

from .common import OcrError, AppContext, load_app_context, ensure_windows_64_bit, configure_stdout_utf8
from .runtime_cache import prepare_local_runtime_dir
from .image import get_image_bytes

__all__ = [
    'OcrError',
    'AppContext',
    'load_app_context',
    'ensure_windows_64_bit',
    'configure_stdout_utf8',
    'prepare_local_runtime_dir',
    'get_image_bytes',
]
