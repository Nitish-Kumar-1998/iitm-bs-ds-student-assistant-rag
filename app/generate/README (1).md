# `app/generate/` — Answer Generation

Turns retrieved chunks into a grounded, cited, streamed answer via Groq.

---

## What it does

| File | What happens |
|------|---------------|
| `prompts.yaml` | Versioned system prompts — strict grounding rules, citation format |
| `generator.py` | Loads prompt, builds context block, calls Groq (streaming), model fallback |
| `run.py` | Chains retrieve + generate into one `ask(query)` function for the API stage |

---

## Design principles (baked into prompts.yaml v1)

1. **Strict grounding** — the model may ONLY use the provided context. No
   general knowledge, no filling gaps with "what seems likely true" about
   IIT Madras or universities in general.

2. **Precision over paraphrase** — exact numbers, dates, fees, and credit
   counts must be quoted exactly as written in the source, never rounded
   or vaguely approximated.

3. **Mandatory citations** — every factual claim must trace back to a
   numbered source. The model lists which sources it actually used at
   the end of every answer.

4. **Conflict handling** — if two sources disagree, the model says so
   explicitly rather than silently picking one.

5. **Honest "I don't know"** — a single, consistent fallback message
   when the answer isn't in context, pointing to the official programme
   page instead of leaving the student stuck or guessing.

---

## Run

**Test generation alone (requires retrieve stage working):**
```bash
python -m app.generate.generator "what is the OPPE exam"
```

**Test the full pipeline (retrieve + generate, what api/ will use):**
```bash
python -m app.generate.run "what is the OPPE exam"
python -m app.generate.run                              # default test query
```

**Validate config:**
```bash
python -m app.generate.config
```

**Use in code (api/ stage will do this):**
```python
from app.generate.run import ask

answer_stream, sources = ask("what is the OPPE exam")
for token in answer_stream:
    print(token, end="", flush=True)

for s in sources:
    print(f"[{s['index']}] {s['doc_title']} — {s['source_url']}")
```

---

## Prompt versioning

Prompts live in `prompts.yaml`, not hardcoded in Python — this lets you
iterate on wording without touching code, and keeps a record of what
changed and why.

**To add a new prompt version:**
1. Copy the `v1` block in `prompts.yaml`
2. Rename the key to `v2`
3. Edit the `system` field
4. Write clear `notes` explaining the change and reasoning
5. Test it: temporarily set `ACTIVE_PROMPT_VERSION = "v2"` in `config.py`
6. Once happy, leave `v2` as the active version (or revert to `v1`)

Old versions are never deleted — this file is your prompt engineering
changelog.

---

## Configuration

| Setting | Value | Why |
|---------|-------|-----|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Strong instruction-following, validated in v1 pipeline |
| `GROQ_FALLBACK_MODELS` | `llama-3.1-8b-instant`, `gemma2-9b-it` | Tried in order if primary model fails |
| `TEMPERATURE` | 0.1 | Low — grounded consistency over creativity |
| `MAX_TOKENS` | 1024 | Generous for detailed policy answers |
| `STREAM_RESPONSE` | True | Token-by-token streaming |
| `CONTEXT_CHUNK_COUNT` | 5 | Matches `RERANK_TOP_K` in retrieve stage |
| `MAX_CHUNK_CHARS` | 2000 | Per-chunk guard against context budget blowout |

---

## Model fallback

If `GROQ_MODEL` errors (rate limit, transient API issue), `generator.py`
automatically retries with each model in `GROQ_FALLBACK_MODELS` in order
before raising. This keeps the assistant available during a partial Groq
outage rather than failing the whole request.

---

## Prerequisites

1. Retrieve stage must be working: `python -m app.retrieve.run "test"`
2. `.env` must have `GROQ_API_KEY` set

---

## Output shape

`ask(query)` returns `(answer, sources)`:

- `answer` — a generator yielding `str` tokens (streaming) or a full `str`
  (non-streaming), depending on the `stream` argument
- `sources` — list of dicts for citation rendering:
  ```json
  [
    {
      "index": 1,
      "doc_title": "OPPE System Compatibility Test - RULES",
      "breadcrumb": "BS-DS May 2026 Grading Document (Student) > OPPE System Compatibility Test - RULES",
      "source_url": "https://docs.google.com/...",
      "chunk_type": "text"
    }
  ]
  ```

The `api/` stage will stream `answer` directly to the client and render
`sources` as citation links.
