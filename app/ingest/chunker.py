"""
chunker.py
----------
Stage 1 of the ingest pipeline.

Reads all markdown files produced by the scraper (app/scraper/data/docs/)
and converts them into a flat list of chunks ready for embedding.

INPUT:
    app/scraper/data/docs/root/     — one .md per root Google Doc
    app/scraper/data/docs/nested/   — one .md per nested Google Doc
    app/scraper/data/registry.json  — metadata index (optional, used for enrichment)

OUTPUT:
    app/ingest/data/chunks.json     — list of chunk dicts

Chunk types produced:
    text    — paragraph / prose content
    table   — markdown table converted to readable plain text
    image   — OCR text extracted from images inline in the doc

Every chunk carries rich metadata sourced from:
    1. YAML frontmatter  — doc_id, title, breadcrumb, depth, root info
    2. HTML section comments — per-section doc/section/breadcrumb/url
    3. Content itself   — heading, heading_level, token_count

Run standalone:
    python -m app.ingest.chunker

Or as part of the full pipeline:
    python -m app.ingest.run
"""

import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingest.config import (
    DOCS_DIR,
    REGISTRY_FILE,
    OUTPUT_DIR,
    CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_TOKENS,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("ingest.chunker")


# =============================================================================
# CONSTANTS
# =============================================================================

# One token ≈ 4 characters — used for fast token estimation without a tokenizer.
# This is intentionally approximate; exact tokenization is not needed here
# because chunk boundaries are soft limits, not hard API constraints.
CHARS_PER_TOKEN = 4

# The scraper embeds per-section metadata as HTML comments in this format:
# <!-- doc:Title | section:Name | breadcrumb:A > B > C | url:https://... -->
_SECTION_COMMENT_RE = re.compile(
    r'<!--\s*doc:(?P<doc>[^|]+)\|'
    r'\s*section:(?P<section>[^|]+)\|'
    r'\s*breadcrumb:(?P<breadcrumb>[^|]+)\|'
    r'\s*url:(?P<url>[^-]+?)\s*-->'
)

# Image blocks written by the scraper:
# [IMAGE: filename.png]
# **OCR Text:** extracted text here
# **Section:** section name
# **Breadcrumb:** A > B > C
_IMAGE_BLOCK_RE = re.compile(
    r'\[IMAGE:\s*(?P<filename>[^\]]+)\]\s*\n'
    r'(?:\*\*OCR Text:\*\*\s*(?P<ocr_text>.*?)\n)?'
    r'(?:\*\*Section:\*\*\s*(?P<section>[^\n]*)\n)?'
    r'(?:\*\*Breadcrumb:\*\*\s*(?P<breadcrumb>[^\n]*))?',
    re.DOTALL,
)

# Table annotation written by the scraper above every table:
# > *Table from section: **Section Name***
_TABLE_ANNOTATION_RE = re.compile(
    r'>\s*\*Table from section:\s*\*\*([^*]+)\*\*\*'
)


# =============================================================================
# UTILITIES
# =============================================================================

def token_count(text: str) -> int:
    """
    Estimate token count from character count.
    1 token ≈ 4 characters (OpenAI/Gemini rule of thumb).
    Fast enough to call on every chunk without overhead.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def make_chunk_id(content: str) -> str:
    """
    Stable MD5 hash of content, truncated to 12 hex chars.

    Why MD5 and not a sequential integer?
    Sequential IDs break when docs are re-scraped and chunk order shifts.
    Content-based IDs are stable: the same text always produces the same ID,
    which means re-ingesting unchanged docs does not create duplicate chunks.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# FRONTMATTER PARSER
# =============================================================================

def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Extract YAML frontmatter from a markdown file.

    The scraper writes frontmatter between --- delimiters at the top of
    every output file. This carries doc-level metadata that applies to
    all chunks from that file.

    Known issue handled here:
    Some document titles contain a bare colon, e.g.:
        title: IITM BS DS to MG Switch Logic : FAQ - for 25f3,26f1
    YAML interprets the second colon as a new mapping key and fails to
    parse the whole block. We detect this and fall back to a line-by-line
    parser that treats everything after the FIRST colon as the value —
    safe because every frontmatter line here is "key: value", never nested.

    Returns:
        (metadata_dict, body_without_frontmatter)

    If no frontmatter is found, returns ({}, raw) so callers always get
    a consistent return type and can fall back gracefully.
    """
    match = re.match(r'^---\n(.*?)\n---\n', raw, re.DOTALL)
    if not match:
        logger.warning("No YAML frontmatter found — falling back to empty metadata")
        return {}, raw

    fm_block = match.group(1)

    try:
        metadata = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError as e:
        logger.debug(f"YAML parse failed, falling back to line parser: {e}")
        metadata = _parse_frontmatter_lines(fm_block)

    body = raw[match.end():]
    return metadata, body


def _parse_frontmatter_lines(fm_block: str) -> dict:
    """
    Fallback line-by-line frontmatter parser for blocks that break strict YAML
    (typically a title or field value containing an unquoted colon).

    Every line is assumed to be "key: value" — we split on the FIRST colon
    only, so values containing additional colons are preserved intact.
    This matches the scraper's actual frontmatter format (flat key-value
    pairs, no nested structures), so it is safe for this specific use case.
    """
    metadata: dict = {}
    for line in fm_block.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip()
    return metadata


# =============================================================================
# SECTION COMMENT PARSER
# =============================================================================

def parse_section_comment(line: str) -> dict | None:
    """
    Parse a scraper-injected HTML comment that marks the start of a section.

    Example comment:
        <!-- doc:FAQ | section:Entry Routes | breadcrumb:FAQ > Entry Routes | url:https://... -->

    Returns a dict with keys: doc, section, breadcrumb, url
    Returns None if the line does not match the expected pattern.

    These comments are how the scraper preserves per-section context
    through the HTML → markdown conversion. We use them to set accurate
    metadata on each chunk rather than inferring it from heading text alone.
    """
    match = _SECTION_COMMENT_RE.search(line)
    if not match:
        return None
    return {
        "doc":        match.group("doc").strip(),
        "section":    match.group("section").strip(),
        "breadcrumb": match.group("breadcrumb").strip(),
        "url":        match.group("url").strip(),
    }


# =============================================================================
# CHUNK BUILDER
# =============================================================================

def build_chunk(
    content:       str,
    chunk_type:    str,
    heading:       str,
    heading_level: int,
    doc_meta:      dict,   # from YAML frontmatter
    section_meta:  dict,   # from HTML comment (overrides doc_meta where set)
) -> dict:
    """
    Assemble a single chunk dict from content + metadata.

    Metadata priority (highest → lowest):
        1. section_meta  — per-section HTML comment (most specific)
        2. doc_meta      — YAML frontmatter (doc-level)
        3. heading text  — fallback if both are absent

    The embed_text field is what gets sent to the embedding model.
    It combines heading + breadcrumb + content so the vector captures
    both the topic context and the actual content — richer than content alone.
    """
    content = content.strip()

    # Resolve metadata with priority: section > frontmatter > heading fallback
    doc_title  = section_meta.get("doc")        or doc_meta.get("title", "")
    section    = section_meta.get("section")    or heading
    breadcrumb = section_meta.get("breadcrumb") or doc_meta.get("breadcrumb", doc_title)
    source_url = section_meta.get("url")        or doc_meta.get("source_url", "")

    # embed_text is richer than content alone:
    # heading gives topic signal, breadcrumb gives location signal
    embed_parts = [p for p in [heading, breadcrumb, content] if p]
    embed_text  = ". ".join(embed_parts)

    return {
        # Identity
        "chunk_id":       make_chunk_id(content),
        "chunk_type":     chunk_type,

        # Content
        "content":        content,
        "embed_text":     embed_text,

        # Location in document
        "heading":        heading,
        "heading_level":  heading_level,
        "section":        section,
        "breadcrumb":     breadcrumb,

        # Source document metadata (from YAML frontmatter)
        "doc_id":         doc_meta.get("doc_id", ""),
        "doc_title":      doc_title,
        "source_url":     source_url,
        "parent_doc_id":  doc_meta.get("parent_doc_id", ""),
        "parent_doc_title": doc_meta.get("parent_doc_title", ""),
        "root_doc_id":    doc_meta.get("root_doc_id", ""),
        "root_doc_title": doc_meta.get("root_doc_title", ""),
        "depth":          doc_meta.get("depth", 0),

        # Pipeline metadata
        "token_count":    token_count(content),
        "created_at":     now_iso(),
        "version":        "2",   # v2 = new scraper + new ingest pipeline
    }


# =============================================================================
# TABLE CONVERTER
# =============================================================================

def table_to_plain_text(table: str) -> str:
    """
    Convert a markdown pipe table to readable plain-text sentences.

    Why convert?
    Cross-encoder rerankers (ms-marco-MiniLM) score pipe-formatted tables
    very poorly because they were trained on natural language, not markdown
    syntax. Converting to "Header: Value" pairs dramatically improves
    reranker scores on table-heavy content like fee tables and milestones.

    Input:
        | Level      | Credits | Total  |
        | ---        | ---     | ---    |
        | Foundation | 32      | 48,000 |

    Output:
        Level | Credits | Total
        Level: Foundation, Credits: 32, Total: 48,000
    """
    lines = [l.strip() for l in table.splitlines() if l.strip()]

    # Keep only pipe rows, skip separator rows (---|---)
    rows = [l for l in lines if l.startswith("|") and not re.match(r'^\|[\s\-|]+\|$', l)]

    if not rows:
        return table  # not a valid table — return as-is

    # Parse each row into cells, stripping leading/trailing pipes and whitespace
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        cells = [c for c in cells if c]  # drop empty cells
        if cells:
            parsed.append(cells)

    if len(parsed) < 2:
        return table  # need at least header + one data row

    header      = parsed[0]
    result_lines = [" | ".join(header)]  # first line: column names

    for data_row in parsed[1:]:
        pairs = []
        for i, value in enumerate(data_row):
            label = header[i] if i < len(header) else f"col{i}"
            pairs.append(f"{label}: {value}")
        result_lines.append(", ".join(pairs))

    return "\n".join(result_lines)


# =============================================================================
# IMAGE BLOCK EXTRACTOR
# =============================================================================

def extract_image_chunks(
    body:          str,
    doc_meta:      dict,
    current_heading:       str,
    current_heading_level: int,
    current_section_meta:  dict,
) -> list[dict]:
    """
    Find all [IMAGE: ...] blocks in the document body and build image chunks.

    The scraper writes image blocks in this format:
        [IMAGE: filename.png]
        **OCR Text:** extracted text here
        **Section:** section name
        **Breadcrumb:** A > B > C

    We prefer OCR text as content. If OCR is absent (image was decorative
    or OCR failed), we skip the chunk — an empty image chunk adds noise.

    Returns a list of image chunks (may be empty).
    """
    chunks = []

    for match in _IMAGE_BLOCK_RE.finditer(body):
        filename  = (match.group("filename") or "").strip()
        ocr_text  = (match.group("ocr_text")  or "").strip()
        section   = (match.group("section")   or "").strip()
        breadcrumb = (match.group("breadcrumb") or "").strip()

        # Skip if no OCR content — decorative images add noise to retrieval
        if not ocr_text:
            logger.debug(f"Skipping image with no OCR text: {filename}")
            continue

        content = f"Image: {filename}\n{ocr_text}"

        # Image blocks carry their own section/breadcrumb written by the scraper
        # Use those if present, fall back to the current section context
        image_section_meta = {
            "doc":        current_section_meta.get("doc", doc_meta.get("title", "")),
            "section":    section    or current_section_meta.get("section", current_heading),
            "breadcrumb": breadcrumb or current_section_meta.get("breadcrumb", ""),
            "url":        current_section_meta.get("url", doc_meta.get("source_url", "")),
        }

        chunk = build_chunk(
            content       = content,
            chunk_type    = "image",
            heading       = current_heading,
            heading_level = current_heading_level,
            doc_meta      = doc_meta,
            section_meta  = image_section_meta,
        )
        chunks.append(chunk)
        logger.debug(f"Image chunk built: {filename} ({token_count(ocr_text)} tokens)")

    return chunks


# =============================================================================
# DOCUMENT CHUNKER
# =============================================================================

def chunk_document(filepath: Path, doc_meta: dict) -> list[dict]:
    """
    Process one markdown file into a list of chunks.

    Two-pass strategy:
        Pass 1 — MarkdownHeaderTextSplitter (semantic boundaries)
            Split at ##, ###, #### headers so each chunk stays on one topic.
            A chunk about "Fee Payment" will not bleed into "Exam Rules".

        Pass 2 — RecursiveCharacterTextSplitter (size enforcement)
            Some sections are very long. This enforces CHUNK_SIZE so no chunk
            exceeds the embedding model's sweet spot. Splits on paragraph
            boundaries first (\n\n), then sentences (\n), then words.

    Special handling:
        Tables  — detected and kept whole (never split mid-table).
                  Converted to plain text for reranker compatibility.
        Images  — extracted separately from [IMAGE: ...] blocks.
                  Each image becomes its own chunk with OCR content.

    Args:
        filepath — path to the .md file
        doc_meta — frontmatter metadata dict (already parsed by caller)

    Returns:
        list of chunk dicts (may be empty if file has no usable content)
    """
    raw = filepath.read_text(encoding="utf-8")

    # Strip frontmatter — doc_meta already carries that information
    _, body = parse_frontmatter(raw)

    chunks            = []
    seen_ids          = set()   # deduplicate within this document
    current_heading        = doc_meta.get("title", filepath.stem)
    current_heading_level  = 1
    current_section_meta   = {}

    # Split body into lines for sequential processing
    lines = body.split("\n")

    # We process the document line by line, accumulating content between
    # section boundaries (headings / section comments).
    # When we hit a new heading or section comment, we flush the accumulated
    # content as one or more chunks.

    pending_lines        = []   # lines accumulated since last flush
    pending_table_lines  = []   # lines of an in-progress table block
    in_table             = False

    def flush_text(lines_to_flush: list[str]) -> None:
        """
        Flush accumulated text lines as chunk(s).
        Applies two-pass splitting: semantic (heading boundary already done
        by the caller) + size enforcement via RecursiveCharacterTextSplitter.
        """
        text = "\n".join(lines_to_flush).strip()
        if not text or token_count(text) < MIN_CHUNK_TOKENS:
            return

        # Remove image blocks from text — they are handled separately
        text_no_images = _IMAGE_BLOCK_RE.sub("", text).strip()
        if not text_no_images or token_count(text_no_images) < MIN_CHUNK_TOKENS:
            return

        # Size enforcement: split long sections into bounded chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = CHUNK_SIZE * CHARS_PER_TOKEN,   # convert tokens → chars
            chunk_overlap = CHUNK_OVERLAP * CHARS_PER_TOKEN,
            separators    = ["\n\n", "\n", ". ", " ", ""],
        )
        sub_texts = splitter.split_text(text_no_images)

        for sub in sub_texts:
            sub = sub.strip()
            if not sub or token_count(sub) < MIN_CHUNK_TOKENS:
                continue

            chunk = build_chunk(
                content       = sub,
                chunk_type    = "text",
                heading       = current_heading,
                heading_level = current_heading_level,
                doc_meta      = doc_meta,
                section_meta  = current_section_meta,
            )

            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                chunks.append(chunk)

    def flush_table(table_lines: list[str]) -> None:
        """
        Flush an accumulated table block as a single table chunk.
        Tables are never split — splitting mid-table destroys meaning.
        """
        raw_table = "\n".join(table_lines).strip()
        if not raw_table:
            return

        # Strip the scraper's annotation line before converting
        # e.g. "> *Table from section: **Fee Structure***"
        raw_table = _TABLE_ANNOTATION_RE.sub("", raw_table).strip()
        if not raw_table:
            return

        plain = table_to_plain_text(raw_table)
        if not plain or token_count(plain) < MIN_CHUNK_TOKENS:
            return

        chunk = build_chunk(
            content       = plain,
            chunk_type    = "table",
            heading       = current_heading,
            heading_level = current_heading_level,
            doc_meta      = doc_meta,
            section_meta  = current_section_meta,
        )

        if chunk["chunk_id"] not in seen_ids:
            seen_ids.add(chunk["chunk_id"])
            chunks.append(chunk)

    for line in lines:
        # ── Section comment ───────────────────────────────────────────────
        # These appear before each section in the scraper output.
        # They carry richer metadata than we can infer from headings alone.
        section_meta = parse_section_comment(line)
        if section_meta:
            current_section_meta = section_meta
            continue  # comment line itself is not content

        # ── Heading ───────────────────────────────────────────────────────
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            # Flush whatever we've been accumulating before moving to new section
            if in_table:
                flush_table(pending_table_lines)
                pending_table_lines = []
                in_table = False
            else:
                flush_text(pending_lines)

            pending_lines         = []
            current_heading       = heading_match.group(2).strip()
            current_heading_level = len(heading_match.group(1))
            continue

        # ── Table detection ───────────────────────────────────────────────
        # Tables start with a pipe character or a blockquote annotation.
        # Once we enter a table, we accumulate until a non-table line appears.
        is_table_line = (
            line.strip().startswith("|")
            or line.strip().startswith("> *Table from section")
        )

        if is_table_line:
            if not in_table:
                # Flush any pending text before starting the table
                flush_text(pending_lines)
                pending_lines = []
                in_table = True
            pending_table_lines.append(line)
            continue

        if in_table and not is_table_line:
            # Non-table line after table — table has ended
            flush_table(pending_table_lines)
            pending_table_lines = []
            in_table = False
            # Fall through to accumulate this line as text

        # ── Regular text line ─────────────────────────────────────────────
        pending_lines.append(line)

    # Flush whatever remains at end of file
    if in_table:
        flush_table(pending_table_lines)
    else:
        flush_text(pending_lines)

    # ── Image chunks ──────────────────────────────────────────────────────
    # Extract all image blocks from the full body in one pass.
    # We pass the last known heading/section context as fallback metadata.
    image_chunks = extract_image_chunks(
        body                  = body,
        doc_meta              = doc_meta,
        current_heading       = current_heading,
        current_heading_level = current_heading_level,
        current_section_meta  = current_section_meta,
    )
    for img_chunk in image_chunks:
        if img_chunk["chunk_id"] not in seen_ids:
            seen_ids.add(img_chunk["chunk_id"])
            chunks.append(img_chunk)

    return chunks


# =============================================================================
# REGISTRY LOADER
# =============================================================================

def load_registry() -> dict[str, dict]:
    """
    Load registry.json as a dict keyed by doc_id.

    The registry is written by the scraper and contains richer metadata
    than what fits in YAML frontmatter (e.g. has_images, has_tables,
    word_count, scraped_at). We use it to enrich doc_meta where available.

    Returns an empty dict if the registry file does not exist — callers
    fall back to frontmatter metadata only.
    """
    if not REGISTRY_FILE.exists():
        logger.warning(f"Registry not found at {REGISTRY_FILE} — using frontmatter only")
        return {}

    with open(REGISTRY_FILE, encoding="utf-8") as f:
        records = json.load(f)

    return {r["doc_id"]: r for r in records if "doc_id" in r}


# =============================================================================
# MAIN
# =============================================================================

def run() -> list[dict]:
    """
    Entry point for Stage 1: Chunker.

    Discovers all markdown files under DOCS_DIR (root/ and nested/),
    processes each one into chunks, deduplicates globally, and saves
    the result to CHUNKS_FILE.

    Returns the final list of chunks (used by run.py to chain stages).
    """
    print("\n" + "═" * 65)
    print("  Stage 1 — Chunker")
    print(f"  Reading from: {DOCS_DIR}")
    print(f"  Writing to:   {CHUNKS_FILE}")
    print("═" * 65)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load registry for metadata enrichment
    registry = load_registry()
    print(f"\n  Registry: {len(registry)} documents indexed")

    # Discover all markdown files (root docs + nested docs)
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    if not md_files:
        print(f"\n  ❌ No markdown files found under {DOCS_DIR}")
        print(f"     Run the scraper first: python -m app.scraper.scraper")
        raise SystemExit(1)

    print(f"  Markdown files found: {len(md_files)}\n")

    all_chunks = []
    seen_ids   = set()   # global deduplication across all documents

    for md_path in md_files:
        # Parse frontmatter to get doc_meta for this file
        raw = md_path.read_text(encoding="utf-8")
        doc_meta, _ = parse_frontmatter(raw)

        # Enrich doc_meta from registry if we have a matching record
        doc_id = doc_meta.get("doc_id", "")
        if doc_id and doc_id in registry:
            registry_record = registry[doc_id]
            # Registry fields take precedence only where frontmatter is empty
            for key in ("title", "source_url", "breadcrumb", "depth",
                        "root_doc_id", "root_doc_title",
                        "parent_doc_id", "parent_doc_title"):
                if not doc_meta.get(key) and registry_record.get(key):
                    doc_meta[key] = registry_record[key]

        doc_title = doc_meta.get("title", md_path.stem)

        # Process the document into chunks
        doc_chunks = chunk_document(md_path, doc_meta)

        # Global deduplication — same content may appear in multiple docs
        # (e.g. a section repeated in both root and nested doc)
        unique_chunks = []
        for chunk in doc_chunks:
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                unique_chunks.append(chunk)

        duplicates = len(doc_chunks) - len(unique_chunks)
        all_chunks.extend(unique_chunks)

        # Per-file summary
        type_counts = {}
        for c in unique_chunks:
            type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1

        dup_note = f"  ({duplicates} duplicates removed)" if duplicates else ""
        print(f"  📄 {md_path.name}")
        print(f"     {doc_title}")
        print(f"     → {len(unique_chunks)} chunks {type_counts}{dup_note}")

    # ── Summary ───────────────────────────────────────────────────────────
    total      = len(all_chunks)
    type_totals: dict[str, int] = {}
    for c in all_chunks:
        t = c["chunk_type"]
        type_totals[t] = type_totals.get(t, 0) + 1

    avg_tokens = sum(c["token_count"] for c in all_chunks) // max(total, 1)

    print(f"\n  {'─' * 40}")
    print(f"  Total chunks:     {total}")
    for chunk_type, count in sorted(type_totals.items()):
        print(f"    {chunk_type:<10}: {count}")
    print(f"  Avg tokens/chunk: {avg_tokens}")

    # ── Validate required fields ──────────────────────────────────────────
    required_fields = [
        "chunk_id", "chunk_type", "content", "embed_text",
        "source_url", "breadcrumb", "created_at", "version",
    ]
    issues = 0
    for chunk in all_chunks:
        for field in required_fields:
            if not chunk.get(field):
                logger.warning(f"Missing '{field}' in chunk {chunk.get('chunk_id', '?')}")
                issues += 1

    if issues:
        print(f"\n  ⚠️  Validation issues: {issues} (see logs)")
    else:
        print(f"\n  ✅ All chunks valid")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"  Saved → {CHUNKS_FILE}")
    print("═" * 65)

    return all_chunks


if __name__ == "__main__":
    from app.ingest.config import validate
    validate()
    run()