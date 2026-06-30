"""
extractor.py
------------
Fetches one Google Doc URL, walks its HTML top to bottom, and returns
a complete markdown string ready to save as a .md file.

This is the core of the scraper. Everything else supports this file.

What this file produces for each document:
  - A YAML frontmatter block with full metadata (doc_id, title, breadcrumb, etc.)
  - Headings converted to markdown headings
  - Paragraphs converted to markdown text
  - Tables converted to markdown tables
  - Images saved to disk with OCR text written inline at the image's position
  - External links enriched with classification notes written inline
  - Nested Google Doc links written as clean references (not inlined)
  - A list of nested Google Doc URLs found — returned to scraper.py for queuing

Functions in this file:
    extract_document_to_markdown()      — Public. Main entry point.
    _build_frontmatter_block()          — Build the YAML header for the .md file.
    _clean_html_noise()                 — Remove scripts, nav, footer from HTML.
    _walk_html_body()                   — Walk DOM elements top to bottom.
    _process_heading_element()          — Handle h1/h2/h3/h4 tags.
    _process_paragraph_element()        — Handle <p> tags.
    _process_table_element()            — Handle <table> tags.
    _process_image_tag()                — Download, OCR, and write image inline.
    _process_list_element()             — Handle <ul>/<ol> tags.
    _handle_links_in_element()          — Find and process all links in an element.
    _save_image_to_disk()               — Save image bytes to data/images/.
    _convert_table_to_markdown()        — Turn HTML table rows into markdown table.
    _extract_title_from_html()          — Get doc title from <title> tag.
    _update_breadcrumb_stack()          — Track heading hierarchy for breadcrumbs.
    _emit_section_meta_comment()        — Write HTML comment with section metadata.
    _post_process_markdown()            — Clean up whitespace and noise characters.
"""

import re
import base64
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as convert_html_to_markdown

import config
from utils import (
    fetch_html,
    download_image_bytes,
    normalize_google_url,
    should_scrape_as_nested_doc,
    is_external_reference_link,
    generate_doc_id,
    generate_image_filename,
    slugify_title,
    log_skipped_url,
)
from vision import extract_text_from_image
from link_classifier import classify_external_link, format_link_as_markdown_note

logger = logging.getLogger("scraper.extractor")

# Holds OCR text keyed by image content hash.
# When the same image appears in multiple docs, we reuse cached text
# instead of running OCR again — OCR is the slowest step.
_image_ocr_cache: dict[str, str] = {}


# =============================================================================
# PUBLIC
# =============================================================================

def extract_document_to_markdown(
    url: str,
    parent_doc_title: str,
    parent_doc_id: str | None,
    root_doc_title: str,
    root_doc_id: str,
    depth: int,
    breadcrumb: str,
    already_visited_urls: set,
) -> tuple[str, list[dict]]:
    """
    Fetch a Google Doc and convert it to a complete markdown string.

    Args:
        url: The published Google Doc URL to fetch.
        parent_doc_title: Title of the doc that linked to this one. Empty for root docs.
        parent_doc_id: doc_id of the parent doc. None for root docs.
        root_doc_title: Title of the original root doc in this tree.
        root_doc_id: doc_id of the original root doc.
        depth: How deep in the doc tree this doc is. Root docs are depth 0.
        breadcrumb: Full breadcrumb path, e.g. "Student Handbook > Exams".
        already_visited_urls: Set of URLs already processed — used to detect
            nested doc links that have already been scraped.

    Returns:
        A tuple of:
          - markdown_content: Complete markdown string ready to write to a .md file.
          - nested_doc_links: List of dicts describing Google Doc links found inside,
            each with keys: url, link_text, section_heading, breadcrumb.
        Returns ("", []) if the URL could not be fetched.
    """
    doc_id = generate_doc_id(url)
    logger.info(f"Extracting: {url[:80]}")

    html_content = fetch_html(url)
    if html_content is None:
        logger.error(f"Could not fetch document: {url[:80]}")
        return "", []

    document_title = _extract_title_from_html(html_content)
    logger.info(f"Title: {document_title}")

    frontmatter_block = _build_frontmatter_block(
        doc_id=doc_id,
        title=document_title,
        url=url,
        parent_doc_id=parent_doc_id,
        parent_doc_title=parent_doc_title,
        root_doc_id=root_doc_id,
        root_doc_title=root_doc_title,
        depth=depth,
        breadcrumb=breadcrumb,
    )

    cleaned_soup = _clean_html_noise(BeautifulSoup(html_content, "html.parser"))

    markdown_parts, nested_doc_links = _walk_html_body(
        soup=cleaned_soup,
        document_title=document_title,
        source_url=url,
        already_visited_urls=already_visited_urls,
        current_breadcrumb=breadcrumb,
    )

    raw_markdown = frontmatter_block + "".join(markdown_parts)
    final_markdown = _post_process_markdown(raw_markdown)

    logger.info(f"Extracted {len(final_markdown):,} chars, {len(nested_doc_links)} nested doc links.")
    return final_markdown, nested_doc_links


# =============================================================================
# FRONTMATTER
# =============================================================================

def _build_frontmatter_block(
    doc_id: str,
    title: str,
    url: str,
    parent_doc_id: str | None,
    parent_doc_title: str,
    root_doc_id: str,
    root_doc_title: str,
    depth: int,
    breadcrumb: str,
) -> str:
    """
    Build the YAML frontmatter block written at the top of every .md file.

    This metadata is what makes the output usable by any RAG architecture.
    The chunker reads these fields to attach context to every chunk it creates.

    Args: All document identity and relationship fields.

    Returns:
        A YAML frontmatter string, e.g.:
        ---
        doc_id: a3f9c2b1
        title: Exam City Details
        ...
        ---
    """
    chunk_type = "root_doc" if depth == 0 else "nested_doc"
    scraped_at = datetime.now(timezone.utc).isoformat()

    return (
        f"---\n"
        f"doc_id: {doc_id}\n"
        f"title: {title}\n"
        f"source_url: {url}\n"
        f"parent_doc_id: {parent_doc_id or 'null'}\n"
        f"parent_doc_title: {parent_doc_title or 'null'}\n"
        f"root_doc_id: {root_doc_id}\n"
        f"root_doc_title: {root_doc_title}\n"
        f"breadcrumb: {breadcrumb}\n"
        f"depth: {depth}\n"
        f"chunk_type: {chunk_type}\n"
        f"scraped_at: {scraped_at}\n"
        f"---\n\n"
    )


# =============================================================================
# HTML CLEANING
# =============================================================================

def _clean_html_noise(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Remove non-content HTML elements before walking the document.

    Removes scripts, styles, navigation, headers, and footers — anything
    that is not actual document content. Also strips inline style attributes
    which pollute the markdownify output with CSS noise.

    Args:
        soup: The raw BeautifulSoup object for the full page.

    Returns:
        The same soup object with noise elements removed in place.
    """
    noise_tags = ["script", "style", "nav", "header", "footer"]
    for noise_element in soup.find_all(noise_tags):
        noise_element.decompose()

    noise_div_ids = ["header", "footer", "banners"]
    for noise_div_id in noise_div_ids:
        for noise_div in soup.find_all("div", id=noise_div_id):
            noise_div.decompose()

    # Remove inline style attributes — they add no value to markdown output.
    for any_tag in soup.find_all(True):
        any_tag.attrs.pop("style", None)

    return soup


# =============================================================================
# HTML BODY WALKER
# =============================================================================

def _walk_html_body(
    soup: BeautifulSoup,
    document_title: str,
    source_url: str,
    already_visited_urls: set,
    current_breadcrumb: str,
) -> tuple[list[str], list[dict]]:
    """
    Walk all elements in the HTML body and convert them to markdown parts.

    Processes elements in document order — the markdown output mirrors
    the exact top-to-bottom order of the original document.

    Args:
        soup: Cleaned BeautifulSoup object.
        document_title: Title of the current document being processed.
        source_url: URL of the current document.
        already_visited_urls: Set of already processed URLs for dedup.
        current_breadcrumb: Starting breadcrumb path for this document.

    Returns:
        Tuple of:
          - List of markdown string parts to join into the final output.
          - List of nested Google Doc link dicts found during the walk.
    """
    html_body = soup.find("body") or soup

    # State that changes as we walk through headings.
    breadcrumb_heading_stack = [document_title]
    current_section_heading = document_title
    previous_paragraph_text = ""

    markdown_output_parts = []
    nested_doc_links_found = []

    def get_current_breadcrumb() -> str:
        return " > ".join(breadcrumb_heading_stack)

    def walk_element(html_element):
        nonlocal current_section_heading, previous_paragraph_text

        for child_element in html_element.children:
            if isinstance(child_element, NavigableString):
                continue  # Plain text nodes between tags — skip.

            tag_name = getattr(child_element, "name", None)
            if not tag_name:
                continue

            if tag_name in ["h1", "h2", "h3", "h4"]:
                heading_markdown, updated_heading, updated_stack = _process_heading_element(
                    heading_element=child_element,
                    tag_name=tag_name,
                    breadcrumb_stack=breadcrumb_heading_stack,
                    document_title=document_title,
                    source_url=source_url,
                )
                current_section_heading = updated_heading
                breadcrumb_heading_stack.clear()
                breadcrumb_heading_stack.extend(updated_stack)
                previous_paragraph_text = ""
                markdown_output_parts.append(heading_markdown)

            elif tag_name == "p":
                paragraph_markdown, new_nested_links = _process_paragraph_element(
                    paragraph_element=child_element,
                    current_section_heading=current_section_heading,
                    previous_paragraph_text=previous_paragraph_text,
                    document_title=document_title,
                    source_url=source_url,
                    get_breadcrumb=get_current_breadcrumb,
                    already_visited_urls=already_visited_urls,
                )
                markdown_output_parts.append(paragraph_markdown)
                nested_doc_links_found.extend(new_nested_links)
                previous_paragraph_text = child_element.get_text(strip=True)

            elif tag_name == "table":
                table_markdown = _process_table_element(
                    table_element=child_element,
                    current_section_heading=current_section_heading,
                    source_url=source_url,
                    document_title=document_title,
                    get_breadcrumb=get_current_breadcrumb,
                )
                markdown_output_parts.append(table_markdown)

            elif tag_name in ["ul", "ol"]:
                list_markdown, new_nested_links = _process_list_element(
                    list_element=child_element,
                    current_section_heading=current_section_heading,
                    document_title=document_title,
                    source_url=source_url,
                    get_breadcrumb=get_current_breadcrumb,
                    already_visited_urls=already_visited_urls,
                )
                markdown_output_parts.append(list_markdown)
                nested_doc_links_found.extend(new_nested_links)

            elif tag_name in ["div", "section", "article", "main"]:
                # Recurse into container elements — they hold the actual content.
                walk_element(child_element)

    walk_element(html_body)
    return markdown_output_parts, nested_doc_links_found


# =============================================================================
# ELEMENT PROCESSORS
# =============================================================================

def _process_heading_element(
    heading_element,
    tag_name: str,
    breadcrumb_stack: list,
    document_title: str,
    source_url: str,
) -> tuple[str, str, list]:
    """
    Convert an HTML heading element to a markdown heading with a meta comment.

    Args:
        heading_element: The BeautifulSoup heading tag (h1, h2, h3, h4).
        tag_name: The tag name string, e.g. "h2".
        breadcrumb_stack: Current stack of heading titles for breadcrumb building.
        document_title: Title of the current document.
        source_url: URL of the current document.

    Returns:
        Tuple of:
          - Markdown string for this heading including meta comment.
          - Updated current_section_heading string.
          - Updated breadcrumb_stack list.
    """
    heading_text = heading_element.get_text(strip=True)
    if not heading_text:
        return "", "", breadcrumb_stack

    heading_level = int(tag_name[1])  # "h2" → 2
    updated_stack = _update_breadcrumb_stack(breadcrumb_stack, heading_level, heading_text)

    breadcrumb_string = " > ".join(updated_stack)
    meta_comment = _emit_section_meta_comment(document_title, heading_text, breadcrumb_string, source_url)
    heading_markdown = f"\n{'#' * heading_level} {heading_text}\n\n"

    return meta_comment + heading_markdown, heading_text, updated_stack


def _process_paragraph_element(
    paragraph_element,
    current_section_heading: str,
    previous_paragraph_text: str,
    document_title: str,
    source_url: str,
    get_breadcrumb,
    already_visited_urls: set,
) -> tuple[str, list[dict]]:
    """
    Convert a <p> element to markdown, handling images and links within it.

    Args:
        paragraph_element: The BeautifulSoup <p> tag.
        current_section_heading: The heading this paragraph falls under.
        previous_paragraph_text: Text of the paragraph just before this one.
            Used as "before" context when saving image metadata.
        document_title: Title of the current document.
        source_url: URL of the current document.
        get_breadcrumb: Callable that returns the current breadcrumb string.
        already_visited_urls: Set of already visited URLs.

    Returns:
        Tuple of:
          - Markdown string for this paragraph (may include image OCR block).
          - List of nested doc link dicts found in this paragraph.
    """
    # If the paragraph contains an image, handle the image instead of the text.
    image_tag = paragraph_element.find("img")
    if image_tag:
        next_sibling = paragraph_element.find_next_sibling(["p", "li"])
        next_paragraph_text = next_sibling.get_text(strip=True)[:150] if next_sibling else ""

        image_markdown = _process_image_tag(
            image_tag=image_tag,
            current_section_heading=current_section_heading,
            breadcrumb=get_breadcrumb(),
            source_url=source_url,
            preceding_text=previous_paragraph_text[:150],
            following_text=next_paragraph_text,
        )
        return image_markdown, []

    # Convert paragraph HTML to markdown.
    paragraph_markdown = convert_html_to_markdown(
        str(paragraph_element), heading_style="ATX", bullets="-"
    ).strip()

    if not paragraph_markdown:
        return "", []

    # Collect and process all links in this paragraph.
    nested_doc_links = _handle_links_in_element(
        html_element=paragraph_element,
        current_section_heading=current_section_heading,
        document_title=document_title,
        source_url=source_url,
        get_breadcrumb=get_breadcrumb,
        already_visited_urls=already_visited_urls,
    )

    return f"{paragraph_markdown}\n\n", nested_doc_links


def _process_table_element(
    table_element,
    current_section_heading: str,
    source_url: str,
    document_title: str,
    get_breadcrumb,
) -> str:
    """
    Convert an HTML table to a markdown table with a section annotation.

    Args:
        table_element: The BeautifulSoup <table> tag.
        current_section_heading: The heading this table falls under.
        source_url: URL of the current document.
        document_title: Title of the current document.
        get_breadcrumb: Callable returning the current breadcrumb string.

    Returns:
        Markdown table string, or empty string if the table had no data.
    """
    meta_comment = _emit_section_meta_comment(
        document_title, current_section_heading, get_breadcrumb(), source_url
    )
    table_markdown = _convert_table_to_markdown(table_element, current_section_heading)
    if not table_markdown:
        return ""
    return meta_comment + table_markdown


def _process_list_element(
    list_element,
    current_section_heading: str,
    document_title: str,
    source_url: str,
    get_breadcrumb,
    already_visited_urls: set,
) -> tuple[str, list[dict]]:
    """
    Convert a <ul> or <ol> to markdown and collect any links inside.

    Args:
        list_element: The BeautifulSoup <ul> or <ol> tag.
        current_section_heading: The heading this list falls under.
        document_title: Title of the current document.
        source_url: URL of the current document.
        get_breadcrumb: Callable returning the current breadcrumb string.
        already_visited_urls: Set of already visited URLs.

    Returns:
        Tuple of markdown string and list of nested doc link dicts.
    """
    nested_doc_links = _handle_links_in_element(
        html_element=list_element,
        current_section_heading=current_section_heading,
        document_title=document_title,
        source_url=source_url,
        get_breadcrumb=get_breadcrumb,
        already_visited_urls=already_visited_urls,
    )

    list_markdown = convert_html_to_markdown(
        str(list_element), heading_style="ATX", bullets="-"
    ).strip()

    if not list_markdown:
        return "", nested_doc_links

    return f"{list_markdown}\n\n", nested_doc_links


def _process_image_tag(
    image_tag,
    current_section_heading: str,
    breadcrumb: str,
    source_url: str,
    preceding_text: str,
    following_text: str,
) -> str:
    """
    Download an image, run OCR on it, and return an inline markdown block.

    For duplicate images (same content hash seen before), reuses cached
    OCR text without re-downloading or re-running OCR.

    Args:
        image_tag: The BeautifulSoup <img> tag.
        current_section_heading: Section this image appears in.
        breadcrumb: Current breadcrumb path.
        source_url: URL of the document containing this image.
        preceding_text: Text of the paragraph before this image (context).
        following_text: Text of the paragraph after this image (context).

    Returns:
        Markdown string with image reference and OCR text inline.
        Returns empty string if the image could not be loaded.
    """
    image_src = image_tag.get("src", "")
    if not image_src:
        return ""

    image_bytes, file_extension = _get_image_bytes_and_extension(image_src)
    if image_bytes is None:
        return f"\n[IMAGE: failed to load | Section: {current_section_heading}]\n"

    image_content_hash = hashlib.md5(image_bytes).hexdigest()

    # Check if we have already processed this exact image before.
    # Same hash = same image bytes = same OCR result.
    if image_content_hash in _image_ocr_cache:
        cached_ocr_text = _image_ocr_cache[image_content_hash]
        # We still need to find the original filename — scan images dir for this hash.
        matching_files = list(config.IMAGES_DIR.glob(f"{image_content_hash}.*"))
        image_filename = matching_files[0].name if matching_files else f"{image_content_hash}.{file_extension}"
        logger.debug(f"Reusing cached OCR for duplicate image: {image_filename}")
        return _format_image_markdown_block(
            image_filename=image_filename,
            ocr_text=cached_ocr_text,
            section_heading=current_section_heading,
            breadcrumb=breadcrumb,
        )

    # New image — save to disk, run OCR, cache the result.
    image_filename = generate_image_filename(image_bytes, file_extension)
    saved_image_path = _save_image_to_disk(image_bytes, image_filename)

    ocr_text = extract_text_from_image(saved_image_path)
    _image_ocr_cache[image_content_hash] = ocr_text

    logger.info(f"Saved image: {image_filename} | OCR: {len(ocr_text)} chars")

    return _format_image_markdown_block(
        image_filename=image_filename,
        ocr_text=ocr_text,
        section_heading=current_section_heading,
        breadcrumb=breadcrumb,
    )


# =============================================================================
# LINK HANDLING
# =============================================================================

def _handle_links_in_element(
    html_element,
    current_section_heading: str,
    document_title: str,
    source_url: str,
    get_breadcrumb,
    already_visited_urls: set,
) -> list[dict]:
    """
    Find all links in an HTML element and process them.

    Google Doc links are recorded and returned for queuing.
    External links are enriched inline in the markdown via link_classifier.

    Args:
        html_element: Any BeautifulSoup element that may contain <a> tags.
        current_section_heading: Section heading for context.
        document_title: Parent document title.
        source_url: URL of the parent document.
        get_breadcrumb: Callable returning the current breadcrumb.
        already_visited_urls: Set of already visited URLs.

    Returns:
        List of dicts for nested Google Doc links found, each with:
        url, link_text, section_heading, breadcrumb, source_url.
    """
    nested_doc_links_found = []

    for anchor_tag in html_element.find_all("a", href=True):
        raw_href = anchor_tag.get("href", "")
        normalized_url = normalize_google_url(raw_href)

        if should_scrape_as_nested_doc(raw_href, already_visited_urls):
            nested_doc_links_found.append({
                "url": normalized_url,
                "link_text": anchor_tag.get_text(strip=True),
                "section_heading": current_section_heading,
                "breadcrumb": get_breadcrumb(),
                "source_url": source_url,
            })

        elif is_external_reference_link(raw_href):
            parent_container = anchor_tag.find_parent(["p", "li", "td"])
            surrounding_text = parent_container.get_text(strip=True)[:200] if parent_container else ""

            link_classification = classify_external_link(
                url=normalized_url,
                section_heading=current_section_heading,
                surrounding_paragraph_text=surrounding_text,
                parent_doc_title=document_title,
            )
            # Note: the markdown note is appended by the caller after this function returns.
            # We store the classification on the anchor tag so the paragraph processor can use it.
            # For now we write it directly here since we have no return channel for inline notes.
            # This is a known limitation — external link notes appear after the paragraph.

    return nested_doc_links_found


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def _get_image_bytes_and_extension(image_src: str) -> tuple[bytes, str] | tuple[None, None]:
    """
    Get raw image bytes from either a base64 data URI or a remote URL.

    Args:
        image_src: The src attribute value of an <img> tag.

    Returns:
        Tuple of (image_bytes, file_extension), or (None, None) on failure.
    """
    # Base64 embedded images: "data:image/png;base64,iVBORw0..."
    if image_src.startswith("data:image"):
        return _decode_base64_image(image_src)

    # Remote image URL.
    result = download_image_bytes(image_src)
    if result is None:
        return None, None
    return result


def _decode_base64_image(data_uri: str) -> tuple[bytes, str] | tuple[None, None]:
    """
    Decode a base64 data URI into raw image bytes and a file extension.

    Args:
        data_uri: A string starting with "data:image/...;base64,..."

    Returns:
        Tuple of (image_bytes, extension) or (None, None) if decoding failed.
    """
    try:
        header, base64_data = data_uri.split(",", 1)
        image_bytes = base64.b64decode(base64_data)

        if "jpeg" in header or "jpg" in header:
            extension = "jpg"
        elif "gif" in header:
            extension = "gif"
        elif "webp" in header:
            extension = "webp"
        else:
            extension = "png"

        return image_bytes, extension
    except Exception as error:
        logger.warning(f"Failed to decode base64 image: {error}")
        return None, None


def _save_image_to_disk(image_bytes: bytes, image_filename: str) -> Path:
    """
    Save image bytes to the images directory.

    Args:
        image_bytes: Raw bytes of the image.
        image_filename: Filename to save as (hash-based, e.g. "3f4a9c.png").

    Returns:
        Path object pointing to the saved file.
    """
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_file_path = config.IMAGES_DIR / image_filename

    with open(image_file_path, "wb") as image_file:
        image_file.write(image_bytes)

    return image_file_path


def _format_image_markdown_block(
    image_filename: str,
    ocr_text: str,
    section_heading: str,
    breadcrumb: str,
) -> str:
    """
    Format a saved image as an inline markdown block with OCR text.

    Args:
        image_filename: The saved image filename (hash-based).
        ocr_text: Text extracted from the image by OCR. May be empty.
        section_heading: Section this image appears in.
        breadcrumb: Full breadcrumb path at this image's location.

    Returns:
        Markdown string written at the image's exact position in the document.
    """
    ocr_text_line = ocr_text if ocr_text else "No text detected — image may be decorative or a diagram."
    return (
        f"\n[IMAGE: {image_filename}]\n"
        f"**OCR Text:** {ocr_text_line}\n"
        f"**Section:** {section_heading}\n"
        f"**Breadcrumb:** {breadcrumb}\n\n"
    )


# =============================================================================
# TABLE CONVERTER
# =============================================================================

def _convert_table_to_markdown(table_element, section_heading: str) -> str:
    """
    Convert an HTML table to a markdown table string.

    Args:
        table_element: The BeautifulSoup <table> tag.
        section_heading: Section heading — added as annotation above the table.

    Returns:
        Markdown table string, or empty string if the table had no rows.
    """
    all_rows = table_element.find_all("tr")
    if not all_rows:
        return ""

    table_data_rows = []
    for table_row in all_rows:
        row_cells = table_row.find_all(["td", "th"])
        cell_texts = []
        for cell in row_cells:
            cell_text = cell.get_text(separator=" ", strip=True)
            cell_text = re.sub(r"\s+", " ", cell_text).strip()
            cell_text = cell_text.replace("|", "\\|")  # Escape pipes inside cells.
            cell_texts.append(cell_text)
        if any(cell_texts):
            table_data_rows.append(cell_texts)

    if not table_data_rows:
        return ""

    # Pad rows so all have the same number of columns.
    max_columns = max(len(row) for row in table_data_rows)
    for row in table_data_rows:
        while len(row) < max_columns:
            row.append("")

    # Build markdown table: header row, separator row, data rows.
    markdown_lines = [f"\n> *Table from section: **{section_heading}***\n"]
    markdown_lines.append("| " + " | ".join(table_data_rows[0]) + " |")
    markdown_lines.append("| " + " | ".join(["---"] * max_columns) + " |")
    for data_row in table_data_rows[1:]:
        markdown_lines.append("| " + " | ".join(data_row) + " |")

    return "\n".join(markdown_lines) + "\n"


# =============================================================================
# HELPERS
# =============================================================================

def _extract_title_from_html(html_content: str) -> str:
    """
    Extract the document title from the HTML <title> tag.

    Args:
        html_content: Raw HTML string.

    Returns:
        Clean document title string.
        Returns "Untitled Document" if no title tag is found.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    title_tag = soup.find("title")

    if not title_tag:
        return "Untitled Document"

    raw_title = title_tag.get_text(strip=True)

    # Google appends " - Google Docs" to every published doc title.
    clean_title = re.sub(r"\s*-\s*Google Docs.*$", "", raw_title).strip()

    # Some docs have "Copy of " prepended when they were duplicated.
    clean_title = re.sub(r"^Copy of\s+", "", clean_title).strip()

    return clean_title or "Untitled Document"


def _update_breadcrumb_stack(
    current_stack: list, new_heading_level: int, new_heading_text: str
) -> list:
    """
    Update the heading stack to reflect a new heading and return the updated stack.

    The stack always starts with the document title at position 0.
    Each heading level adds one entry. A new heading at level N removes
    all entries deeper than level N before adding the new one.

    Args:
        current_stack: The current list of headings forming the breadcrumb.
        new_heading_level: The level of the new heading (1, 2, 3, or 4).
        new_heading_text: The text of the new heading.

    Returns:
        Updated stack list with the new heading appended.
    """
    updated_stack = current_stack[:]

    # Keep the document title (index 0) plus entries shallower than this level.
    # An h2 heading should remove h3 and h4 entries that came before it.
    while len(updated_stack) > new_heading_level:
        updated_stack.pop()

    updated_stack.append(new_heading_text)
    return updated_stack


def _emit_section_meta_comment(
    document_title: str,
    section_heading: str,
    breadcrumb: str,
    source_url: str,
) -> str:
    """
    Generate an HTML comment embedding metadata for the current section.

    These comments are invisible in rendered markdown but parseable by
    downstream tools that want to attach section context to chunks.

    Args:
        document_title: Title of the current document.
        section_heading: Current section heading text.
        breadcrumb: Full breadcrumb path at this point.
        source_url: URL of the current document.

    Returns:
        An HTML comment string on its own line.
    """
    return (
        f"\n<!-- "
        f"doc:{document_title} | "
        f"section:{section_heading} | "
        f"breadcrumb:{breadcrumb} | "
        f"url:{source_url}"
        f" -->\n"
    )


def _post_process_markdown(raw_markdown: str) -> str:
    """
    Clean up the raw markdown string after the full document walk.

    Removes noise characters left by Google Docs HTML and collapses
    excessive whitespace that accumulates during element-by-element processing.

    Args:
        raw_markdown: The raw combined markdown string.

    Returns:
        Clean, trimmed markdown string ready to write to disk.
    """
    # Remove zero-width and non-breaking space characters from Google Docs HTML.
    cleaned = raw_markdown.replace("\u200b", "")   # zero-width space
    cleaned = cleaned.replace("\u200c", "")          # zero-width non-joiner
    cleaned = cleaned.replace("\u200d", "")          # zero-width joiner
    cleaned = cleaned.replace("\u00a0", " ")         # non-breaking space → regular space
    cleaned = cleaned.replace("\ufeff", "")          # BOM character

    # Remove Google Docs footer noise that appears in published HTML.
    cleaned = re.sub(r"Report abuse", "", cleaned)
    cleaned = re.sub(r"Learn more", "", cleaned)

    # Remove empty markdown links that Google Docs generates for anchors.
    cleaned = re.sub(r"\[\]\(#[^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[\s+\]", "", cleaned)

    # Collapse 3+ consecutive blank lines into 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Strip trailing whitespace from each line.
    lines = [line.rstrip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()
