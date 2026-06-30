"""
utils.py
--------
Shared helper functions used by extractor.py and scraper.py.
No scraping or parsing logic lives here — only reusable plumbing.

Functions in this file:
    fetch_html()                    — Get HTML from a URL safely
    download_image_bytes()          — Download raw image bytes from a URL
    normalize_google_url()          — Clean messy Google URLs into one canonical form
    is_google_doc_url()             — Check if a URL is a Google Doc
    is_blocked_domain()             — Check if a URL is in the skip list
    should_scrape_as_nested_doc()   — Final decision: should we follow this link?
    is_external_reference_link()    — Is this an external link worth classifying?
    generate_doc_id()               — Stable 8-char ID for a document URL
    generate_image_filename()       — Stable filename for an image based on its content
    slugify_title()                 — Turn a doc title into a safe filename
    load_checkpoint()               — Load previously visited URLs to resume a run
    save_checkpoint()               — Save visited URLs so a crash does not lose progress
    log_skipped_url()               — Record a skipped URL and reason to skipped.log
    wait_between_requests()         — Polite delay between HTTP requests
"""

import re
import json
import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from config import (
    REQUEST_TIMEOUT_SECONDS,
    DELAY_BETWEEN_REQUESTS_SECONDS,
    SKIPPED_LOG,
    CHECKPOINT_FILE,
    GOOGLE_DOC_URL_PATTERN,
    SKIP_DOMAINS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("scraper.utils")


# =============================================================================
# HTTP
# =============================================================================

def fetch_html(url: str) -> str | None:
    """
    Fetch the HTML content of a URL.

    Args:
        url: The full URL to fetch.

    Returns:
        HTML string on success. None if blocked, login-walled, or failed.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Scraper/2.0)"}
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers=headers)

        if response.status_code == 401:
            log_skipped_url(url, reason="HTTP 401 — requires authentication")
            return None

        if response.status_code != 200:
            log_skipped_url(url, reason=f"HTTP {response.status_code}")
            return None

        # Google returns HTTP 200 for login-walled docs but shows a sign-in page.
        # We detect this by checking for sign-in indicators in the page content.
        if _is_google_login_page(response.text):
            log_skipped_url(url, reason="Login wall — document requires IITM sign-in")
            return None

        return response.text

    except requests.exceptions.Timeout:
        log_skipped_url(url, reason=f"Timeout after {REQUEST_TIMEOUT_SECONDS}s")
    except requests.exceptions.ConnectionError:
        log_skipped_url(url, reason="Connection error")
    except Exception as error:
        log_skipped_url(url, reason=f"Unexpected error: {str(error)[:60]}")

    return None


def download_image_bytes(image_url: str) -> tuple[bytes, str] | None:
    """
    Download raw bytes of an image from a URL.

    Args:
        image_url: Direct URL to the image file.

    Returns:
        Tuple of (raw_bytes, file_extension) on success.
        Extension is one of: "png", "jpg", "gif", "webp".
        Returns None if the download failed.
    """
    try:
        response = requests.get(image_url, timeout=REQUEST_TIMEOUT_SECONDS)

        if response.status_code != 200:
            logger.warning(f"Image download failed HTTP {response.status_code}: {image_url[:80]}")
            return None

        file_extension = _get_extension_from_content_type(
            response.headers.get("Content-Type", "image/png")
        )
        return response.content, file_extension

    except Exception as error:
        logger.warning(f"Image download error: {error} | {image_url[:80]}")
        return None


def _is_google_login_page(html_content: str) -> bool:
    """Return True if the HTML is a Google sign-in page, not actual doc content."""
    login_indicators = ["Sign in", "accounts.google.com/signin"]
    return any(indicator in html_content for indicator in login_indicators)


def _get_extension_from_content_type(content_type: str) -> str:
    """Convert an HTTP Content-Type header to a file extension string."""
    content_type = content_type.lower()
    if "jpeg" in content_type or "jpg" in content_type:
        return "jpg"
    if "gif" in content_type:
        return "gif"
    if "webp" in content_type:
        return "webp"
    return "png"  # Safe default for unknown image types.


# =============================================================================
# URL NORMALIZATION AND CLASSIFICATION
# =============================================================================

def normalize_google_url(raw_url: str) -> str:
    """
    Normalize any Google URL variant into one canonical form.

    The same Google Doc can appear as many different URL shapes in hrefs.
    Normalizing collapses them all so deduplication works correctly.

    Args:
        raw_url: Any raw URL string found in an href attribute.

    Returns:
        A clean canonical URL string. Empty string if input was empty.
    """
    if not raw_url:
        return ""

    url = raw_url.strip()

    # Unwrap Google redirect: google.com/url?q=ACTUAL_URL
    if "google.com/url" in url and "?q=" in url:
        try:
            url = unquote(url.split("?q=")[1].split("&")[0])
        except (IndexError, ValueError):
            pass

    # Drop anchor fragments — they point to a section within a page, not a new page.
    url = url.split("#")[0]

    # Remove /u/N/ user-scoped prefix: docs.google.com/document/u/2/d/ → /document/d/
    url = re.sub(
        r"docs\.google\.com/document/u/\d+/d/",
        "docs.google.com/document/d/",
        url,
    )

    # Remove /edit, /view, /copy suffixes — they are the same doc, different views.
    url = re.sub(r"/(edit|view|copy)(\?.*)?$", "", url)

    # Remove query strings from non-published docs (/pub URLs keep their query string).
    if "docs.google.com/document" in url and not url.endswith("/pub"):
        url = url.split("?")[0]

    return url.rstrip("/")


def is_google_doc_url(url: str) -> bool:
    """
    Return True if this URL points to a Google Doc.

    Args:
        url: A normalized URL string.

    Returns:
        True if the URL matches the Google Doc pattern.
    """
    return bool(re.search(GOOGLE_DOC_URL_PATTERN, url))


def is_blocked_domain(url: str) -> bool:
    """
    Return True if this URL belongs to a domain we never follow.

    Args:
        url: Any URL string.

    Returns:
        True if the URL contains a domain from the SKIP_DOMAINS list.
    """
    return any(blocked_domain in url for blocked_domain in SKIP_DOMAINS)


def should_scrape_as_nested_doc(raw_url: str, already_visited_urls: set) -> bool:
    """
    Return True if we should fetch this URL and save it as a nested doc.

    A URL is followed only if it passes all of these checks:
      - Not empty or an anchor-only link
      - Not from a blocked domain
      - Is a Google Doc URL
      - Has not already been visited in this run

    Args:
        raw_url: A raw URL string from an href attribute.
        already_visited_urls: Set of URLs already processed in this run.

    Returns:
        True if we should scrape this URL as a nested document.
    """
    if not raw_url or raw_url.startswith("#"):
        return False

    if is_blocked_domain(raw_url):
        return False

    normalized_url = normalize_google_url(raw_url)

    if not is_google_doc_url(normalized_url):
        return False

    if normalized_url in already_visited_urls:
        return False

    return True


def is_external_reference_link(raw_url: str) -> bool:
    """
    Return True if this URL is an external link worth classifying.

    External reference links are non-Google-Doc HTTP links that get
    enriched by link_classifier.py and written as notes in the markdown.

    Args:
        raw_url: A raw URL string from an href attribute.

    Returns:
        True if this is an external link we should classify and note.
    """
    if not raw_url or raw_url.startswith("#"):
        return False

    if is_blocked_domain(raw_url):
        return False

    normalized_url = normalize_google_url(raw_url)

    # Google Docs are handled separately — they are not external references.
    if is_google_doc_url(normalized_url):
        return False

    return normalized_url.startswith("http")


# =============================================================================
# ID AND FILENAME GENERATION
# =============================================================================

def generate_doc_id(url: str) -> str:
    """
    Generate a stable 8-character unique ID for a document URL.

    Uses MD5 of the normalized URL so the same doc always gets the
    same ID across runs, even if scraped months apart.

    Args:
        url: The canonical URL of the document.

    Returns:
        An 8-character hex string, e.g. "a3f9c2b1".
    """
    normalized_url = normalize_google_url(url)
    return hashlib.md5(normalized_url.encode()).hexdigest()[:8]


def generate_image_filename(image_bytes: bytes, file_extension: str) -> str:
    """
    Generate a stable filename for an image based on its content hash.

    Hash-based naming means the same image always gets the same filename,
    across any number of runs. Deduplication is automatic — writing the
    same bytes to the same filename is a no-op.

    Args:
        image_bytes: Raw bytes of the image.
        file_extension: Extension without dot, e.g. "png", "jpg".

    Returns:
        Filename string, e.g. "3f4a9c2b1d8e7f6a.png".
    """
    content_hash = hashlib.md5(image_bytes).hexdigest()
    return f"{content_hash}.{file_extension}"


def slugify_title(title: str) -> str:
    """
    Convert a document title into a safe, readable filename.

    Args:
        title: Document title string, e.g. "Exam City Details (Updated)".

    Returns:
        Safe filename string, e.g. "exam_city_details_updated".
    """
    slug = title.lower()
    slug = re.sub(r"[-\s]+", "_", slug)       # spaces and hyphens to underscores
    slug = re.sub(r"[^\w]", "", slug)          # remove non-alphanumeric characters
    slug = re.sub(r"_+", "_", slug)            # collapse multiple underscores
    slug = slug.strip("_")
    return slug[:60]                            # cap length for filesystem safety


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint() -> set:
    """
    Load the set of already-visited URLs from the checkpoint file.

    Returns:
        Set of URL strings already scraped. Empty set if no checkpoint exists.
    """
    if not CHECKPOINT_FILE.exists():
        logger.info("No checkpoint found — starting a fresh scrape.")
        return set()

    try:
        with open(CHECKPOINT_FILE, encoding="utf-8") as checkpoint_file:
            data = json.load(checkpoint_file)
            visited_urls = set(data.get("visited_urls", []))
            logger.info(f"Checkpoint loaded — resuming with {len(visited_urls)} URLs already done.")
            return visited_urls
    except (json.JSONDecodeError, KeyError) as error:
        logger.warning(f"Checkpoint file corrupted ({error}) — starting fresh.")
        return set()


def save_checkpoint(visited_urls: set) -> None:
    """
    Save the current visited URL set to disk.

    Called after every successfully processed document so progress
    is never lost by more than one document on a crash.

    Args:
        visited_urls: The full set of URLs processed so far in this run.
    """
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as checkpoint_file:
        json.dump({"visited_urls": list(visited_urls)}, checkpoint_file, indent=2)


# =============================================================================
# LOGGING
# =============================================================================

def log_skipped_url(url: str, reason: str) -> None:
    """
    Record a skipped URL and the reason to the skipped log file.

    Args:
        url: The URL that was skipped.
        reason: Human-readable explanation of why it was skipped.
    """
    SKIPPED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SKIPPED_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"SKIPPED | {reason:<50} | {url}\n")
    logger.warning(f"Skipped: {reason} | {url[:80]}")


# =============================================================================
# RATE LIMITING
# =============================================================================

def wait_between_requests() -> None:
    """Pause between HTTP requests to avoid being blocked by Google."""
    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
