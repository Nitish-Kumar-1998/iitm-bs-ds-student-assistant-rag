"""
scraper.py
----------
Entry point for the IITM BS Knowledge Base Scraper.
Run this file to start a scrape: python scraper.py

What this file does:
  1. Creates all output folders inside data/
  2. Loads checkpoint (so a crashed run resumes where it left off)
  3. Scrapes each root document from config.ROOT_DOCS
  4. Finds all nested Google Doc links inside each root doc
  5. Scrapes every nested doc and saves it as its own .md file
  6. Repeats until no more new Google Doc links are found (BFS)
  7. Writes registry.json — an index of every doc scraped
  8. Prints a summary

Output:
  data/docs/root/     — one .md file per root document
  data/docs/nested/   — one .md file per nested document found
  data/images/        — every unique image, named by content hash
  data/registry.json  — full index of all scraped documents
  data/checkpoint.json — resume state (deleted after clean run)
  data/skipped.log    — URLs that failed and why

To add more root docs: edit ROOT_DOCS in config.py. Nothing else changes.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import config
from utils import (
    normalize_google_url,
    generate_doc_id,
    slugify_title,
    load_checkpoint,
    save_checkpoint,
    wait_between_requests,
    log_skipped_url,
)
from extractor import extract_document_to_markdown

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("scraper.main")


# =============================================================================
# REGISTRY
# =============================================================================

# Holds one entry per scraped document, written to registry.json at the end.
# Downstream systems (chunker, embedder) read this to know what was scraped.
scraped_document_registry: list[dict] = []


def add_document_to_registry(
    doc_id: str,
    title: str,
    url: str,
    parent_doc_id: str | None,
    parent_doc_title: str,
    root_doc_id: str,
    root_doc_title: str,
    breadcrumb: str,
    depth: int,
    output_file_path: Path,
    markdown_content: str,
) -> None:
    """
    Add a scraped document's metadata to the registry list.

    Args:
        doc_id: Stable 8-char ID for this document.
        title: Document title.
        url: Source URL.
        parent_doc_id: doc_id of the parent. None for root docs.
        parent_doc_title: Title of the parent. Empty for root docs.
        root_doc_id: doc_id of the original root doc in this tree.
        root_doc_title: Title of the root doc.
        breadcrumb: Full breadcrumb path.
        depth: Nesting depth. Root = 0.
        output_file_path: Path where the .md file was saved.
        markdown_content: The full markdown string (used to compute stats).
    """
    word_count = len(markdown_content.split())
    has_images = "[IMAGE:" in markdown_content
    has_tables = "| --- |" in markdown_content
    has_external_links = "🔗 **External Reference**" in markdown_content

    scraped_document_registry.append({
        "doc_id":           doc_id,
        "title":            title,
        "source_url":       url,
        "parent_doc_id":    parent_doc_id,
        "parent_doc_title": parent_doc_title,
        "root_doc_id":      root_doc_id,
        "root_doc_title":   root_doc_title,
        "breadcrumb":       breadcrumb,
        "depth":            depth,
        "chunk_type":       "root_doc" if depth == 0 else "nested_doc",
        "output_file":      str(output_file_path.relative_to(config.BASE_DIR)),
        "word_count":       word_count,
        "has_images":       has_images,
        "has_tables":       has_tables,
        "has_external_links": has_external_links,
        "scraped_at":       datetime.now(timezone.utc).isoformat(),
    })


def save_registry_to_disk() -> None:
    """Write the full document registry to data/registry.json."""
    with open(config.REGISTRY_FILE, "w", encoding="utf-8") as registry_file:
        json.dump(scraped_document_registry, registry_file, indent=2, ensure_ascii=False)
    logger.info(f"Registry saved: {len(scraped_document_registry)} documents.")


# =============================================================================
# FOLDER SETUP
# =============================================================================

def create_output_folders() -> None:
    """Create all required output folders if they do not already exist."""
    folders_to_create = [
        config.ROOT_DOCS_DIR,
        config.NESTED_DOCS_DIR,
        config.IMAGES_DIR,
    ]
    for folder_path in folders_to_create:
        folder_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output folders ready.")


# =============================================================================
# ROOT DOC SCRAPING
# =============================================================================

def scrape_root_document(
    root_doc: dict,
    already_visited_urls: set,
) -> list[dict]:
    """
    Scrape one root document and save it as a .md file in data/docs/root/.

    Args:
        root_doc: One entry from config.ROOT_DOCS with keys: url, title, filename.
        already_visited_urls: The global visited URL set — updated in place.

    Returns:
        List of nested doc link dicts found inside this root doc.
        Each dict has: url, link_text, section_heading, breadcrumb, source_url.
    """
    normalized_url = normalize_google_url(root_doc["url"])

    if normalized_url in already_visited_urls:
        logger.info(f"Skipping root doc (already done): {root_doc['title']}")
        return []

    already_visited_urls.add(normalized_url)
    doc_id = generate_doc_id(normalized_url)
    output_file_path = config.ROOT_DOCS_DIR / root_doc["filename"]

    logger.info(f"--- Root doc: {root_doc['title']}")

    markdown_content, nested_doc_links = extract_document_to_markdown(
        url=normalized_url,
        parent_doc_title="",
        parent_doc_id=None,
        root_doc_title=root_doc["title"],
        root_doc_id=doc_id,
        depth=0,
        breadcrumb=root_doc["title"],
        already_visited_urls=already_visited_urls,
    )

    if not markdown_content:
        logger.error(f"Root doc extraction failed: {root_doc['title']}")
        return []

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Saved: {output_file_path.name} ({len(markdown_content):,} chars)")

    add_document_to_registry(
        doc_id=doc_id,
        title=root_doc["title"],
        url=normalized_url,
        parent_doc_id=None,
        parent_doc_title="",
        root_doc_id=doc_id,
        root_doc_title=root_doc["title"],
        breadcrumb=root_doc["title"],
        depth=0,
        output_file_path=output_file_path,
        markdown_content=markdown_content,
    )

    save_checkpoint(already_visited_urls)
    wait_between_requests()

    return nested_doc_links


# =============================================================================
# NESTED DOC SCRAPING
# =============================================================================

def scrape_nested_document(
    nested_doc_url: str,
    parent_doc_id: str,
    parent_doc_title: str,
    root_doc_id: str,
    root_doc_title: str,
    parent_breadcrumb: str,
    depth: int,
    already_visited_urls: set,
) -> list[dict]:
    """
    Scrape one nested Google Doc and save it as a .md file in data/docs/nested/.

    Args:
        nested_doc_url: The normalized URL of the nested doc to scrape.
        parent_doc_id: doc_id of the doc that linked to this one.
        parent_doc_title: Title of the doc that linked to this one.
        root_doc_id: doc_id of the original root doc in this tree.
        root_doc_title: Title of the original root doc.
        parent_breadcrumb: Breadcrumb path of the parent doc.
        depth: Nesting depth of this doc (parent depth + 1).
        already_visited_urls: Global visited URL set — updated in place.

    Returns:
        List of further nested doc link dicts found inside this doc.
    """
    if nested_doc_url in already_visited_urls:
        return []

    already_visited_urls.add(nested_doc_url)
    doc_id = generate_doc_id(nested_doc_url)

    logger.info(f"--- Nested doc (depth {depth}): {nested_doc_url[:70]}")

    markdown_content, further_nested_links = extract_document_to_markdown(
        url=nested_doc_url,
        parent_doc_title=parent_doc_title,
        parent_doc_id=parent_doc_id,
        root_doc_title=root_doc_title,
        root_doc_id=root_doc_id,
        depth=depth,
        breadcrumb=f"{parent_breadcrumb} > (nested)",
        already_visited_urls=already_visited_urls,
    )

    if not markdown_content:
        logger.error(f"Nested doc extraction failed: {nested_doc_url[:70]}")
        return []

    # Use a slugified version of the doc title for the filename.
    # We extract the title from the frontmatter for accuracy.
    doc_title = _extract_title_from_markdown_frontmatter(markdown_content)
    safe_filename = f"{slugify_title(doc_title)}_{doc_id}.md"
    output_file_path = config.NESTED_DOCS_DIR / safe_filename

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Saved: {safe_filename} ({len(markdown_content):,} chars)")

    actual_breadcrumb = f"{parent_breadcrumb} > {doc_title}"

    add_document_to_registry(
        doc_id=doc_id,
        title=doc_title,
        url=nested_doc_url,
        parent_doc_id=parent_doc_id,
        parent_doc_title=parent_doc_title,
        root_doc_id=root_doc_id,
        root_doc_title=root_doc_title,
        breadcrumb=actual_breadcrumb,
        depth=depth,
        output_file_path=output_file_path,
        markdown_content=markdown_content,
    )

    save_checkpoint(already_visited_urls)
    wait_between_requests()

    return further_nested_links


# =============================================================================
# BFS ORCHESTRATOR
# =============================================================================

def run_breadth_first_scrape_of_all_nested_docs(
    initial_nested_doc_links: list[dict],
    root_doc_id: str,
    root_doc_title: str,
    root_doc_breadcrumb: str,
    already_visited_urls: set,
) -> None:
    """
    Process all nested Google Doc links using a breadth-first queue.

    Breadth-first means we fully process all docs at depth 1 before
    moving to depth 2, then depth 3, and so on. This gives a more
    predictable and readable log output, and means shallower (more
    authoritative) docs are always scraped before deeper ones.

    Args:
        initial_nested_doc_links: Nested doc links found in the root doc.
        root_doc_id: doc_id of the root doc these links belong to.
        root_doc_title: Title of the root doc.
        root_doc_breadcrumb: Breadcrumb of the root doc (its title).
        already_visited_urls: Global visited URL set — updated in place.
    """
    # Queue holds tuples of:
    # (url, parent_doc_id, parent_doc_title, parent_breadcrumb, depth)
    processing_queue = deque()

    root_doc_parent_id = generate_doc_id(
        [d["url"] for d in config.ROOT_DOCS if d["title"] == root_doc_title][0]
    )

    for nested_link in initial_nested_doc_links:
        processing_queue.append((
            nested_link["url"],
            root_doc_parent_id,
            root_doc_title,
            root_doc_breadcrumb,
            1,  # depth 1 — direct children of a root doc
        ))

    total_processed = 0

    while processing_queue:
        (
            doc_url,
            parent_doc_id,
            parent_doc_title,
            parent_breadcrumb,
            current_depth,
        ) = processing_queue.popleft()

        if doc_url in already_visited_urls:
            continue

        further_nested_links = scrape_nested_document(
            nested_doc_url=doc_url,
            parent_doc_id=parent_doc_id,
            parent_doc_title=parent_doc_title,
            root_doc_id=root_doc_id,
            root_doc_title=root_doc_title,
            parent_breadcrumb=parent_breadcrumb,
            depth=current_depth,
            already_visited_urls=already_visited_urls,
        )

        total_processed += 1

        # Add any newly discovered nested links to the back of the queue.
        this_doc_id = generate_doc_id(doc_url)
        for further_link in further_nested_links:
            if further_link["url"] not in already_visited_urls:
                processing_queue.append((
                    further_link["url"],
                    this_doc_id,
                    further_link.get("link_text", ""),
                    further_link["breadcrumb"],
                    current_depth + 1,
                ))

    logger.info(f"BFS complete — {total_processed} nested docs scraped for root: {root_doc_title}")


# =============================================================================
# HELPERS
# =============================================================================

def _extract_title_from_markdown_frontmatter(markdown_content: str) -> str:
    """
    Extract the title field from the YAML frontmatter of a markdown string.

    Args:
        markdown_content: Full markdown string with frontmatter at the top.

    Returns:
        Title string, or "Untitled Document" if not found.
    """
    for line in markdown_content.splitlines():
        if line.startswith("title:"):
            return line.replace("title:", "").strip()
    return "Untitled Document"


def print_run_summary() -> None:
    """Print a human-readable summary of the completed scrape."""
    root_count = sum(1 for doc in scraped_document_registry if doc["depth"] == 0)
    nested_count = sum(1 for doc in scraped_document_registry if doc["depth"] > 0)
    image_count = len(list(config.IMAGES_DIR.glob("*.*")))
    total_words = sum(doc["word_count"] for doc in scraped_document_registry)

    print("\n" + "═" * 60)
    print("  IITM BS Knowledge Base Scraper — Run Complete")
    print("═" * 60)
    print(f"  Root documents scraped:    {root_count}")
    print(f"  Nested documents scraped:  {nested_count}")
    print(f"  Total documents:           {root_count + nested_count}")
    print(f"  Images saved:              {image_count}")
    print(f"  Total words extracted:     {total_words:,}")
    print(f"\n  Output:")
    print(f"    {config.ROOT_DOCS_DIR}")
    print(f"    {config.NESTED_DOCS_DIR}")
    print(f"    {config.IMAGES_DIR}")
    print(f"    {config.REGISTRY_FILE}")
    print(f"\n  Next step: run the chunker on data/docs/")
    print("═" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    """
    Main entry point — runs the full scraping pipeline from start to finish.

    Execution flow:
      1. Create output folders.
      2. Load checkpoint (resume if crashed).
      3. For each root doc: scrape it, collect nested links.
      4. BFS over all nested links until none remain.
      5. Save registry.json.
      6. Print summary.
    """
    create_output_folders()
    already_visited_urls = load_checkpoint()

    print("\n" + "═" * 60)
    print("  IITM BS Knowledge Base Scraper — Starting")
    print(f"  Root documents to process: {len(config.ROOT_DOCS)}")
    print("═" * 60 + "\n")

    for root_doc in config.ROOT_DOCS:
        nested_doc_links_from_root = scrape_root_document(
            root_doc=root_doc,
            already_visited_urls=already_visited_urls,
        )

        if nested_doc_links_from_root:
            root_doc_id = generate_doc_id(normalize_google_url(root_doc["url"]))
            run_breadth_first_scrape_of_all_nested_docs(
                initial_nested_doc_links=nested_doc_links_from_root,
                root_doc_id=root_doc_id,
                root_doc_title=root_doc["title"],
                root_doc_breadcrumb=root_doc["title"],
                already_visited_urls=already_visited_urls,
            )

    save_registry_to_disk()
    print_run_summary()


if __name__ == "__main__":
    run()
