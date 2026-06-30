# IITM BS Knowledge Base Scraper

Scrapes published Google Docs into clean, structured markdown files
ready for chunking and embedding into a RAG pipeline.

---

## What It Does

Give it Google Doc URLs. It produces one `.md` file per document —
root docs and every nested Google Doc they link to — with:

- All text in reading order
- Tables converted to markdown tables
- Images saved to disk with OCR text written inline at the image's exact position
- External links (forms, sheets, videos) enriched with classification notes
- Rich YAML frontmatter on every file for downstream RAG use

---

## Folder Structure

```
scraper/
├── scraper.py          # Entry point — run this
├── extractor.py        # Core HTML → markdown logic for one document
├── link_classifier.py  # Classifies external links (forms, sheets, videos, portals)
├── vision.py           # OCR — extracts text from a saved image file
├── utils.py            # Shared helpers: fetch, URL cleaning, checkpoint, IDs
├── config.py           # All settings and root doc URLs — edit this to add docs
├── README.md           # This file
└── data/               # All output (created automatically on first run)
    ├── docs/
    │   ├── root/       # One .md per root document
    │   └── nested/     # One .md per nested Google Doc found
    ├── images/         # Every unique image, named by MD5 hash of its bytes
    ├── registry.json   # Index of every document scraped
    ├── checkpoint.json # Resume state — deleted safely after a clean run
    └── skipped.log     # URLs that failed and why
```

---

## Install

```bash
pip install requests beautifulsoup4 markdownify rapidocr
```

Python 3.10 or higher required.

---

## Add Your Root Docs

Open `config.py` and add entries to `ROOT_DOCS`:

```python
ROOT_DOCS = [
    {
        "url": "https://docs.google.com/document/d/e/YOUR_DOC_ID/pub",
        "title": "Your Document Title",
        "filename": "your_output_filename.md",
    },
]
```

Rules:
- URL must be a published Google Doc ending in `/pub`
- `filename` is the output file name in `data/docs/root/` — use `snake_case.md`
- Every nested Google Doc found inside will be saved automatically in `data/docs/nested/`

---

## Run

```bash
cd scraper
python scraper.py
```

Output appears in `data/`. The script prints a summary when done.

**If it crashes mid-run:** just run it again. The checkpoint file resumes from where it stopped.

**To start fresh:** delete `data/checkpoint.json` and run again.

---

## Output Format

### Every .md file starts with YAML frontmatter

```yaml
---
doc_id: a3f9c2b1
title: Exam City Details
source_url: https://docs.google.com/...
parent_doc_id: b7e2a1c4
parent_doc_title: Student Handbook
root_doc_id: b7e2a1c4
root_doc_title: Student Handbook
breadcrumb: Student Handbook > Exams > Exam Cities
depth: 1
chunk_type: nested_doc
scraped_at: 2026-06-22T10:30:00Z
---
```

### Images appear inline at their exact position

```
[IMAGE: 3f4a9c2b1d8e7f6a.png]
**OCR Text:** North Zone: Delhi | South Zone: Chennai
**Section:** Exam Cities
**Breadcrumb:** Student Handbook > Exams > Exam Cities
```

### External links appear as enriched notes

```
> 🔗 **External Reference**
> **Title:** MLP Project Registration Form
> **What it contains:** Google Form to register for the MLP project
> **When to refer:** Student asks about MLP project registration
> **Category:** `registration_form`
> **Access:** public
> **URL:** https://forms.gle/...
```

### Nested Google Doc links appear as clean references

```
[→ See: Exam City Details](https://docs.google.com/...)
<!-- nested_doc_id: a3f9c2b1 -->
```

The linked doc is scraped separately and saved in `data/docs/nested/`.

---

## registry.json

Every scraped document gets one entry:

```json
{
  "doc_id": "a3f9c2b1",
  "title": "Exam City Details",
  "source_url": "https://docs.google.com/...",
  "parent_doc_id": "b7e2a1c4",
  "parent_doc_title": "Student Handbook",
  "root_doc_id": "b7e2a1c4",
  "root_doc_title": "Student Handbook",
  "breadcrumb": "Student Handbook > Exams > Exam Cities",
  "depth": 1,
  "chunk_type": "nested_doc",
  "output_file": "data/docs/nested/exam_city_details_a3f9c2b1.md",
  "word_count": 312,
  "has_images": false,
  "has_tables": true,
  "has_external_links": true,
  "scraped_at": "2026-06-22T10:30:00Z"
}
```

Downstream chunkers and embedders read this file to know what exists and how docs relate to each other.

---

## What Each File Does

| File | Job |
|------|-----|
| `scraper.py` | Entry point. Creates folders, loads checkpoint, loops over root docs, runs BFS over nested docs, saves registry. |
| `extractor.py` | Fetches one URL, walks its HTML top-to-bottom, returns a complete markdown string and list of nested doc links found. |
| `link_classifier.py` | Given an external URL and its context, returns a title, description, and category for it. |
| `vision.py` | Loads RapidOCR engine once, runs OCR on a single image file, returns extracted text. |
| `utils.py` | fetch_html, download_image_bytes, normalize_google_url, generate_doc_id, slugify_title, checkpoint load/save, skip logging. |
| `config.py` | ROOT_DOCS list, all folder paths, network timeouts, log settings. |

---

## Key Functions

### scraper.py
| Function | What it does |
|----------|-------------|
| `run()` | Main entry point — runs the full pipeline |
| `scrape_root_document()` | Scrapes one root doc, saves to data/docs/root/ |
| `scrape_nested_document()` | Scrapes one nested doc, saves to data/docs/nested/ |
| `run_breadth_first_scrape_of_all_nested_docs()` | BFS queue over all nested links |
| `add_document_to_registry()` | Adds one doc's metadata to the registry list |
| `save_registry_to_disk()` | Writes registry.json |

### extractor.py
| Function | What it does |
|----------|-------------|
| `extract_document_to_markdown()` | Main entry — fetch + walk + return markdown |
| `_build_frontmatter_block()` | Build the YAML header |
| `_clean_html_noise()` | Remove scripts, nav, footer from HTML |
| `_walk_html_body()` | Walk DOM elements in document order |
| `_process_heading_element()` | h1/h2/h3/h4 → markdown heading + meta comment |
| `_process_paragraph_element()` | p → markdown text, handles images and links inside |
| `_process_table_element()` | table → markdown table |
| `_process_image_tag()` | Download image, OCR it, write inline block |
| `_process_list_element()` | ul/ol → markdown list |
| `_handle_links_in_element()` | Find all links, record nested docs, classify external links |
| `_convert_table_to_markdown()` | HTML rows/cells → markdown table string |
| `_post_process_markdown()` | Clean whitespace and Google Docs noise characters |

### utils.py
| Function | What it does |
|----------|-------------|
| `fetch_html()` | GET a URL, handle errors and login walls, return HTML or None |
| `download_image_bytes()` | Download image bytes from a URL |
| `normalize_google_url()` | Collapse all Google URL variants to one canonical form |
| `should_scrape_as_nested_doc()` | Final yes/no: follow this link as a Google Doc? |
| `is_external_reference_link()` | Is this a non-Google-Doc link worth classifying? |
| `generate_doc_id()` | MD5 of URL → stable 8-char ID |
| `generate_image_filename()` | MD5 of image bytes → stable filename |
| `slugify_title()` | "Exam City Details" → "exam_city_details" |
| `load_checkpoint()` | Load visited URLs from disk |
| `save_checkpoint()` | Save visited URLs to disk |
| `log_skipped_url()` | Write skipped URL + reason to skipped.log |

### vision.py
| Function | What it does |
|----------|-------------|
| `extract_text_from_image()` | Public. Run OCR on one image file, return text string |
| `_load_ocr_engine()` | Load RapidOCR once, cache it, try both package names |
| `_parse_ocr_result()` | Normalize result shape across RapidOCR versions |

### link_classifier.py
| Function | What it does |
|----------|-------------|
| `classify_external_link()` | Public. Match URL to known patterns, return ExternalLinkInfo |
| `format_link_as_markdown_note()` | Format ExternalLinkInfo as a markdown blockquote |
| `_classify_google_sheets()` | Handle spreadsheet links |
| `_classify_google_forms()` | Handle Google Forms links |
| `_classify_youtube()` | Handle YouTube video links |
| `_classify_kaggle()` | Handle Kaggle competition links |
| `_classify_github()` | Handle GitHub repo links |
| `_classify_iitm_portals()` | Handle official IITM and related portals |
| `_generic_fallback_classification()` | Fallback for any unknown link |

---

## How to Extend

**Add a new root doc:** Add an entry to `ROOT_DOCS` in `config.py`.

**Add a new external link type:** Add an `elif` block in `link_classifier.py` following the same pattern as the existing classifiers.

**Change OCR engine:** Replace `_load_ocr_engine()` in `vision.py`. The rest of the pipeline does not change.

**Change output format:** Edit `_build_frontmatter_block()` and `_format_image_markdown_block()` in `extractor.py`.

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: rapidocr` | `pip install rapidocr` |
| `ModuleNotFoundError: bs4` | `pip install beautifulsoup4` |
| `ModuleNotFoundError: markdownify` | `pip install markdownify` |
| Doc shows login page | The doc requires IITM sign-in. Check skipped.log. |
| Empty output file | The doc was fetched but had no parseable content. Check the raw URL in a browser. |

---

## Next Step After Scraping

Feed `data/docs/` into your chunker.
The YAML frontmatter on every file gives the chunker everything it needs:
`doc_id`, `breadcrumb`, `depth`, `chunk_type`, `root_doc_title`.

Use `data/registry.json` as the index to iterate over all scraped files.
