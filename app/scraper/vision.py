"""
vision.py
---------
Extracts text from a single image file using RapidOCR.
Called by extractor.py each time a new image is saved during scraping.
The returned text is written directly into the markdown at the image's position.

Functions in this file:
    extract_text_from_image()   — Public. OCR one image file, return text string.
    _load_ocr_engine()          — Internal. Load RapidOCR model once, cache it.
    _parse_ocr_result()         — Internal. Normalize result shape across versions.
    _get_text_from_result_item()— Internal. Extract text from one detected region.
"""

import logging
from pathlib import Path

logger = logging.getLogger("scraper.vision")

# Holds the loaded RapidOCR engine after first use.
# None means not loaded yet — loaded lazily on first call to extract_text_from_image().
_cached_ocr_engine = None


# =============================================================================
# PUBLIC
# =============================================================================

def extract_text_from_image(saved_image_path: Path) -> str:
    """
    Extract all readable text from a saved image file using OCR.

    An empty string return means no text was detected — the image is likely
    a logo, decorative divider, or diagram with no text labels.

    Args:
        saved_image_path: Path to the image file on disk.

    Returns:
        All detected text joined by newlines.
        Empty string if no text found or OCR failed.
    """
    ocr_engine = _load_ocr_engine()

    if ocr_engine is None:
        return ""  # Engine failed to load — already logged, fail silently.

    try:
        raw_ocr_result = ocr_engine(str(saved_image_path))
    except Exception as error:
        logger.warning(f"OCR failed on {saved_image_path.name}: {error}")
        return ""

    detected_text_lines = _parse_ocr_result(raw_ocr_result)

    if not detected_text_lines:
        logger.debug(f"No text detected in: {saved_image_path.name}")
        return ""

    return "\n".join(detected_text_lines)


# =============================================================================
# INTERNAL
# =============================================================================

def _load_ocr_engine():
    """
    Load the RapidOCR engine and cache it for reuse.

    Tries the current package name first, then the legacy name.
    Loading is lazy — happens only on first call, not at import time.

    Returns:
        A RapidOCR engine instance, or None if neither package is installed.
    """
    global _cached_ocr_engine

    if _cached_ocr_engine is not None:
        return _cached_ocr_engine

    # Try current package name first (actively maintained).
    try:
        from rapidocr import RapidOCR
        _cached_ocr_engine = RapidOCR()
        logger.info("OCR engine loaded via 'rapidocr'.")
        return _cached_ocr_engine
    except ImportError:
        pass

    # Try legacy package name as fallback.
    try:
        from rapidocr_onnxruntime import RapidOCR
        _cached_ocr_engine = RapidOCR()
        logger.info("OCR engine loaded via 'rapidocr_onnxruntime'.")
        return _cached_ocr_engine
    except ImportError:
        pass

    logger.error(
        "RapidOCR not installed. Images will be saved without OCR text.\n"
        "Fix: pip install rapidocr"
    )
    return None


def _parse_ocr_result(raw_ocr_result) -> list[str]:
    """
    Normalize a raw RapidOCR result into a flat list of text strings.

    Handles two different result shapes returned by different RapidOCR versions:
      - Tuple shape (older):  (list_of_detections, elapsed_time)
      - Object shape (newer): object with .txts attribute

    Args:
        raw_ocr_result: The direct return value from calling the OCR engine.

    Returns:
        List of non-empty text strings, one per detected text region.
    """
    if raw_ocr_result is None:
        return []

    # Older versions return a tuple — extract the detections list.
    if isinstance(raw_ocr_result, tuple):
        raw_ocr_result = raw_ocr_result[0]

    if not raw_ocr_result:
        return []

    # Newer versions return an object with a .txts attribute.
    if hasattr(raw_ocr_result, "txts"):
        return [text.strip() for text in raw_ocr_result.txts if text and text.strip()]

    # Older versions return a list of detection items.
    return [
        text
        for item in raw_ocr_result
        for text in [_get_text_from_result_item(item)]
        if text
    ]


def _get_text_from_result_item(detection_item) -> str:
    """
    Extract the text string from one OCR detection item.

    Args:
        detection_item: One item from the OCR result list.
            Either [bounding_box, text, confidence] or object with .text attribute.

    Returns:
        Detected text string, or empty string if not extractable.
    """
    # List/tuple form: index 1 is always the text string.
    if isinstance(detection_item, (list, tuple)) and len(detection_item) >= 2:
        text = detection_item[1]
        if isinstance(text, str):
            return text.strip()

    # Object form.
    if hasattr(detection_item, "text"):
        return detection_item.text.strip()

    return ""
