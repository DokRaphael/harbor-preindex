# harbor-preindex


`harbor-preindex` is a Python CLI tool for preparing a filesystem for semantic retrieval and assisted file placement.

It helps answer questions such as:

- Given a new file, which existing project folder should it go into?
- Where is my resume?
- Where are my Amazon invoices?
- Where is the code that talks to Qdrant?

It is designed for large, already-existing storage trees such as a NAS, archive, or migration dump, where files are mostly organized but new incoming documents are harder to place consistently.

The tool works by:

1. scanning an existing storage root,
2. building a lightweight semantic profile for each project folder,
3. indexing those profiles in a local vector store,
4. then comparing a new incoming file against that index to suggest the best destination folder.

`harbor-preindex` is the first building block of **Harbor**. It produces stable JSON outputs that can later be consumed by a larger orchestration layer.

---

## What problem does it solve?

As storage grows over time, it becomes harder to know where new files belong.

For example, imagine you receive a file named:

`contract_dupont.pdf`

And your storage already contains folders such as:

- `Clients/Dupont`
- `Archives/Projects_2024`
- `Admin/Contracts`

A human can usually guess the right destination quickly.  
A machine can do something similar if it builds a lightweight semantic representation of the existing folders and compares the new file against them.

That is what `harbor-preindex` does.

It is **not** a full file manager, and it does **not** try to move files automatically across your whole system.

It is a focused **pre-indexing and folder suggestion layer**.

---

## What the current version does

Version 1 is intentionally narrow and practical:

- text-first, with semantic enrichment for documents and code-like text files
- two retrieval levels: existing folder index plus a lightweight file index
- local crawl of an already-mounted storage root
- embeddings via configurable Ollama HTTP backends
- optional LLM-based reranking for incoming-file folder suggestion
- LLM-free hybrid retrieval for plain text queries
- local Qdrant storage
- automatic decision when confidence is high enough
- structured JSON output for later Harbor integration

The goal of this first version is not to solve every modality or every edge case.  
The goal is to build a clean, local-first, extensible base.

---

## How it works

The query pipeline is explicit and built around a signal extraction layer:

```text
file -> SignalExtractor -> ExtractedSignal -> SemanticEnricher -> EnrichedSignal -> embedding -> retrieval -> decision
```

In practice, the workflow looks like this:

1. scan the storage root
2. detect candidate project folders
3. build a lightweight semantic profile for each folder
4. generate embeddings for those profiles
5. store them in local Qdrant
6. extract a signal from a new incoming file
7. retrieve the top matching folders
8. either auto-select the best one or ask for review if confidence is too low

Signal extraction is still intentionally lightweight, but file cards are now enriched with a semantic layer before embedding. This enrichment keeps the pipeline local-first and deterministic while giving retrieval a better compact representation of documents and code-like text files.

The retrieval core now also supports plain text search such as:

- `where is my resume?`
- `where are my Amazon invoices?`
- `find the harbor docs`

There are now two related query paths:

1. file placement
   file -> SignalExtractor -> ExtractedSignal -> SemanticEnricher -> EnrichedSignal -> embedding -> retrieval -> decision

2. semantic retrieval
   text query -> QueryHintExtractor -> StructuredQueryHints -> embedding -> hybrid retrieval -> light hint-aware rerank -> explanation


This text query path uses a simple hybrid search:

1. search the existing folder-level index
2. search a new file-level index
3. extract lightweight structured query hints
4. apply a small hint-aware rerank bonus when hints align with the candidates
5. return a stable JSON response with `match_type`, `confidence`, `needs_review`, and a compact `why` explanation per match

The standard response stays compact. If you want structured explanation details for debugging, use:

```bash
harbor-preindex query --debug-evidence "where is the code that talks to qdrant?"
```

This adds an `evidence` object per match with compact overlap signals such as matched sources, query terms, topic hints, entity candidates, imports, or symbols. It is heuristic by design and meant for explainability, not exact contribution accounting.

It also adds top-level `query_hints`, which exposes the lightweight structured interpretation of the query used for hint-aware reranking and explanation.
These hints are soft signals only. They are not a heavy filter layer and do not hard-exclude results by default.

---

## Why this project exists

Many semantic search systems start only at query time.

`harbor-preindex` moves part of the work earlier:

* prepare a cold storage tree,
* summarize project folders,
* build a reusable local semantic index,
* make later retrieval and routing more reliable.

In other words, this project is not just about searching files.

It is about **preparing a filesystem for semantic retrieval and assisted organization**.

---

## Features

### Indexing

* crawl a local storage root
* detect project folders with a simple heuristic
* build a lightweight text profile per folder
* build a lightweight semantic card per supported file
* enrich file signals with semantic hints before building file cards
* generate embeddings in batches
* store folder and file indexes locally in Qdrant

### Querying a new file

* extract a signal from the incoming file
* retrieve top-k candidate folders
* auto-select when confidence is high enough
* use a compact LLM rerank step when confidence is borderline
* fall back to `review_needed` when ambiguity remains

### Querying the retrieval core

* embed a plain text query
* extract lightweight structured query hints from the query
* retrieve top-k candidate files
* retrieve top-k candidate folders
* apply a small soft rerank bonus when query hints align with candidate semantics
* merge both lists without requiring an LLM or heavy filters
* return a stable JSON response for higher-level Harbor consumers

### Local persistence

* JSON outputs for queries and index runs
* SQLite audit trail
* local persistent Qdrant storage
* structured JSON logs on `stderr`

---

## Architecture

Main modules:

* `harbor_preindex/crawler/` — NAS or filesystem scan and folder sampling
* `harbor_preindex/profiling/` — lightweight text extraction and folder profile building
* `harbor_preindex/signals/` — `SignalExtractor` abstraction and normalized extracted signals
* `harbor_preindex/semantic/` — lightweight semantic enrichment for documents and code-like text files
* `harbor_preindex/embedding/` — `EmbeddingBackend` abstraction and Ollama embedding backend
* `harbor_preindex/llm/` — `LLMBackend` abstraction and Ollama LLM backend
* `harbor_preindex/storage/` — local Qdrant, JSON results, SQLite audit
* `harbor_preindex/retrieval/` — top-k retrieval
* `harbor_preindex/decision/` — confidence gate and LLM reranking

---

## Requirements

* Python 3.11+
* a local Python virtual environment
* read access to the storage root you want to index
* Ollama installed locally and reachable over HTTP
* Ollama models available for:

  * embeddings
  * LLM-based reranking / decision

No separate installation is required for:

* SQLite, because this project uses Python's built-in `sqlite3` module
* Qdrant server, because this MVP uses `qdrant-client` in local persistent mode

Example Ollama endpoint:

```bash
http://localhost:11434
```

---

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

External dependency to install separately.

For example, on macOS with Homebrew:

```bash
brew install ollama
```

Then start Ollama locally:

```bash
ollama serve
```

---

## Quick start

For a first local run, the provided `.env.example` uses project-relative paths and a local Ollama endpoint.

```bash
cp .env.example .env
mkdir -p ./data/storage-root
harbor-preindex health-check
harbor-preindex build-index
harbor-preindex query-file /tmp/incoming/contract_dupont.pdf
harbor-preindex query "where is my resume?"
```

If you use the default models from `.env.example`, make sure Ollama is running locally and that these models are available:

```bash
ollama pull embeddinggemma
ollama pull qwen2.5:7b-instruct
```

Replace `HARBOR_ROOT` with your real archive root when you move from a local test setup to a NAS or server environment.

---

## Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Important variables:

```env
HARBOR_ROOT=./data/storage-root
HARBOR_DATA_DIR=./data/runtime
HARBOR_LOG_DIR=./data/runtime/logs

OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=embeddinggemma
LLM_MODEL=qwen2.5:7b-instruct

OLLAMA_MAX_RETRIES=2
EMBEDDING_BATCH_SIZE=16

QDRANT_MODE=local
QDRANT_COLLECTION=projects
QDRANT_FILE_COLLECTION=files
QDRANT_PATH=./data/runtime/qdrant

TOP_K=5

EXCLUDED_PATH_SEGMENTS=build,dist,node_modules,.git,__pycache__,example,examples,test,tests,testdata,assets,static,libraries
```

Notes:

* configuration is driven only by `.env` and process environment variables
* `.env.example` is intentionally local-first; switch to absolute storage paths when deploying on a NAS or VM
* `HARBOR_ROOT` is the scan root, but is never itself indexed as a candidate project folder
* if `RESULTS_DIR` or `SQLITE_PATH` are empty, they are derived automatically from `HARBOR_DATA_DIR`
* if `HARBOR_LOG_DIR` is not set, it falls back to `HARBOR_DATA_DIR/logs`

---

## CLI commands

### Health check

```bash
harbor-preindex health-check
```

### Build the index

```bash
harbor-preindex build-index
```

Example output:

```json
{
  "collection": "projects",
  "file_collection": "files",
  "generated_at": "2026-04-06T00:00:00Z",
  "indexed_files": 214,
  "indexed_projects": 42,
  "recreated_collection": false,
  "root_path": "/data/storage-root",
  "scanned_directories": 42
}
```

### Rebuild the index from scratch

```bash
harbor-preindex rescan
```

### Query a new incoming file

```bash
harbor-preindex query-file /tmp/incoming/contract_dupont.pdf
```

### Query the retrieval core

```bash
harbor-preindex query "where is my resume?"
```

Example output:

```json
{
  "query": "where is my resume?",
  "match_type": "likely_file",
  "confidence": 0.87,
  "needs_review": false,
  "matches": [
    {
      "target_kind": "file",
      "target_id": "f8d4b5a2-56ad-51fb-bb21-90d72884e8f5",
      "path": "/data/storage-root/admin/cv/Raphael_Dok_CV.txt",
      "score": 0.91,
      "label": "Raphael_Dok_CV.txt",
      "why": "filename and extracted text strongly match query"
    },
    {
      "target_kind": "folder",
      "target_id": "admin_cv",
      "path": "/data/storage-root/admin/cv",
      "score": 0.78,
      "label": "admin/cv",
      "why": "folder path and sample filenames match query"
    }
  ],
  "generated_at": "2026-04-06T00:00:00Z"
}
```

### Optional debug mode

```bash
harbor-preindex query-file --debug-profiles /tmp/incoming/contract_dupont.pdf
```

This adds extra fields such as:

* the extracted text profile for the input file
* the `text_profile` of the top candidate folders

---

### Debug evidence for retrieval queries

```bash
harbor-preindex query --debug-evidence "where is the code that talks to qdrant?"
```

This adds extra fields such as:

* `evidence.matched_sources` for the parts of the file or folder that overlapped with the query
* `evidence.source_terms` for the compact matched terms grouped by source
* `evidence.matched_topic_hints`, `matched_entity_candidates`, `matched_imports`, or `matched_symbols` when available

---

## Example query result

```json
{
  "input_file": "/tmp/incoming/contract_dupont.pdf",
  "top_candidates": [
    {
      "project_id": "clients_dupont",
      "path": "/data/storage-root/Clients/Dupont",
      "score": 0.91
    },
    {
      "project_id": "archives_projects_2024",
      "path": "/data/storage-root/Archives/Projects_2024",
      "score": 0.74
    }
  ],
  "decision": {
    "selected_project_id": "clients_dupont",
    "selected_path": "/data/storage-root/Clients/Dupont",
    "confidence": 0.94,
    "mode": "llm_rerank"
  },
  "generated_at": "2026-04-06T00:00:00Z"
}
```

---

## Qdrant usage

This project uses `qdrant-client` in local persistent mode.

No external Qdrant server or Docker setup is required for this MVP.

SQLite is also local-only in this project, via Python's standard library. No separate database server is required.

Each project folder is stored as one point in the `projects` collection, with metadata such as:

* `project_id`
* `path`
* `name`
* `parent`
* `sample_filenames`
* `doc_count`
* `text_profile`

`project_id` is the human-readable identifier exposed in outputs.

The internal Qdrant `point.id` is a deterministic UUID derived from the folder path, so it remains stable across rescans.

Supported files are also stored as one point per file in the `files` collection, with metadata such as:

* `file_id`
* `path`
* `filename`
* `extension`
* `parent_path`
* `modality`
* `text_for_embedding`

This keeps deployment simple while remaining compatible with a future move to remote Qdrant.

---

## Supported extraction in V1

Current extraction support:

* filenames
* text documents such as `.txt` and `.md`
* text-based PDF via `pypdf`
* code-like text files when their extensions are included in `SUPPORTED_EXTENSIONS`

PDF extraction is best effort.

If a PDF is corrupted, encrypted, or malformed:

* profiling continues with the filename only
* a warning is logged
* PDF extraction successes and failures are reported in application logs during indexing

The semantic enrichment layer does not perform OCR, deep parsing, or mandatory LLM classification. It adds compact hints such as:

* document or code-like kind hints
* topic hints
* entity candidates
* time hints
* structural hints
* a short functional summary

For code-like text files, the current implementation is intentionally lightweight:

* good support for Python-style imports and symbols
* generic fallback heuristics for other code and config-like files
* no full AST pipeline
* no language-specific compiler or parser dependency

---

## Decision logic

The decision engine is intentionally simple:

1. if the top result is very strong and clearly ahead of the second result, the decision is automatic
2. otherwise, a compact LLM receives a minimal context and the top candidates
3. the LLM response must pass strict JSON validation
4. if validation fails or ambiguity remains, the system falls back to `review_needed`

The goal is predictable behavior and safe structured outputs.

In practice, the decision `mode` can be `auto_top1`, `llm_rerank`, or `review_needed`.

---

## Local outputs

By default, runtime data is created under `HARBOR_DATA_DIR`:

* `results/` — query outputs and indexing summaries
* `results/` also stores retrieval query outputs from `harbor-preindex query`
* `harbor-preindex.sqlite3` — local run audit trail
* `qdrant/` — local vector storage

Application logs are structured as JSON on `stderr`, which makes them easy to integrate with `systemd`.

---

## Helper scripts

Simple wrappers are provided in:

* `scripts/build-index.sh`
* `scripts/query.sh`
* `scripts/query-file.sh`
* `scripts/health-check.sh`

A sample `systemd` unit is also provided:

* `scripts/harbor-preindex-build-index.service.example`

---

## Current limitations

Version 1 makes deliberately simple choices:

* project folder detection uses a lightweight heuristic
* semantic profiles and file cards are compact
* file-level retrieval is intentionally lightweight and does not yet do OCR or chunk-level search
* no OCR
* no image, audio, or video extraction yet
* no HTTP API
* no multi-node orchestration
* no full multimodal pipeline yet

This is a focused MVP, not a complete production platform.

---

## Possible next steps

* better handling of structured query hints such as “my plumber invoice from last year”
* lightweight query-time hint extraction without heavy filtering
  Examples:
  `invoice` -> kind hint
  `last year` -> coarse time hint
  `amazon` -> entity hint
  `code that talks to qdrant` -> technical topic hint
* better project-folder detection
* controlled recursive profile enrichment
* local API or daemon mode
* remote Qdrant support
* multi-node Harbor integration (images, videos, audio)
* multimodal extractors (add images, videos, audio support)
* better confidence strategy and human feedback loop

---

## Project positioning

`harbor-preindex` is not just another embedding script.

It is a **semantic pre-indexing layer for filesystems**, designed to prepare an existing storage tree for workflows such as:

* folder suggestion
* semantic retrieval
* assisted classification
* future Harbor orchestration

---

## Status

Experimental MVP.
Local-first, readable, and designed to be extended.
