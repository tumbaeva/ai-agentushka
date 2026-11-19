"""Best-effort OCR for scanned PDF/DOCX without a text layer.

Pure-pip stack: ``rapidocr-onnxruntime`` (OCR engine, models bundled in the
wheel) + ``pypdfium2`` (PDF rasterisation) + ``Pillow``. Everything is imported
lazily, so the application keeps working when these optional extras are not
installed — :func:`ocr_available` simply reports ``False``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

MAX_PAGES = 50
MAX_IMAGES = 50
RENDER_SCALE = 2.0  # PDF zoom for rasterisation

_engine = None

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        import pypdfium2  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
    return _engine


def _ocr_array(image) -> str:
    import numpy as np

    engine = _get_engine()
    result, _ = engine(np.asarray(image))
    if not result:
        return ""
    lines = [item[1] for item in result if item and len(item) > 1 and item[1]]
    return "\n".join(lines)


def ocr_image_bytes(data: bytes) -> str:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return _ocr_array(img.convert("RGB"))


def ocr_pdf(path: Path, max_pages: int = MAX_PAGES, scale: float = RENDER_SCALE) -> str:
    import pypdfium2 as pdfium

    parts: list[str] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(min(len(pdf), max_pages)):
            bitmap = pdf[i].render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            text = _ocr_array(image)
            if text.strip():
                parts.append(text)
    finally:
        pdf.close()
    return "\n\n".join(parts)


def ocr_docx(path: Path, max_images: int = MAX_IMAGES) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        media = sorted(n for n in zf.namelist() if n.startswith("word/media/"))
        for name in media[:max_images]:
            if Path(name).suffix.lower() not in IMAGE_EXT:
                continue
            try:
                text = ocr_image_bytes(zf.read(name))
            except Exception:
                continue
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def ocr_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return ocr_pdf(path)
    if ext == ".docx":
        return ocr_docx(path)
    return ""
