"""
config.py
---------
Single source of truth for all scraper settings, paths, and root document URLs.
No logic lives here — only data and configuration values.

Every other file imports from here.
To change any setting, change it here. Nothing else needs to be touched.
"""

from pathlib import Path


# =============================================================================
# ROOT DOCUMENTS
# =============================================================================
# Add, remove, or edit entries here to control what gets scraped.
# Each entry produces exactly one .md file in data/docs/root/.
#
# Fields:
#   url      — The published Google Doc URL (must end in /pub)
#   title    — Human readable title used in frontmatter and logs
#   filename — Output filename in data/docs/root/ (use snake_case.md)

ROOT_DOCS = [
    {
        "url": (
            "https://docs.google.com/document/d/e/"
            "2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjs"
            "GrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub"
        ),
        "title": "IITM BS Degree Programme - Student Handbook",
        "filename": "student_handbook.md",
    },
    {
        "url": (
            "https://docs.google.com/document/d/e/"
            "2PACX-1vT5PBOz4OH663W0IJPVGVjG_nfmYZGfFI7W1j-6wTLcex13O_7BZmf6a96Q6liO0W-mLZB5hOGZeNNl/pub"
        ),
        "title": "BS-DS May 2026 Grading Document (Student)",
        "filename": "grading_document.md",
    },
    {
        "url": (
            "https://docs.google.com/document/d/e/"
            "2PACX-1vQSbrYHmnAf44ioAIAq1jOGOVl4atWDztEhVL9Yk5uNJc8-5P280W4VUdM4_"
            "eobpAsrn71Ovym04nqV/pub"
        ),
        "title": "Announcement Document for the May Term",
        "filename": "announcement_document.md",
    },
]

# =============================================================================
# PATHS
# =============================================================================
# BASE_DIR is the scraper/ folder (where this file lives).
# All paths are relative to BASE_DIR so the project is fully portable.

BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
ROOT_DOCS_DIR   = DATA_DIR / "docs" / "root"
NESTED_DOCS_DIR = DATA_DIR / "docs" / "nested"
IMAGES_DIR      = DATA_DIR / "images"

REGISTRY_FILE   = DATA_DIR / "registry.json"
CHECKPOINT_FILE = DATA_DIR / "checkpoint.json"
SKIPPED_LOG     = DATA_DIR / "skipped.log"


# =============================================================================
# LINK RULES
# =============================================================================
# Controls which links are followed as Google Docs vs treated as external refs.

# Any href matching this pattern is scraped as a nested Google Doc.
GOOGLE_DOC_URL_PATTERN = r"docs\.google\.com/document/(?:d|u/\d+/d)/[^/]+"

# These domains are never followed — they are noise (footer, help, sign-in links).
SKIP_DOMAINS = [
    "accounts.google.com",
    "support.google.com",
    "google.com/intl",
    "policies.google.com",
    "workspace.google.com",
    
]


# =============================================================================
# NETWORK
# =============================================================================

REQUEST_TIMEOUT_SECONDS        = 20   # Give up on a single request after this many seconds
DELAY_BETWEEN_REQUESTS_SECONDS = 0.8  # Pause between requests to avoid being blocked


# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
