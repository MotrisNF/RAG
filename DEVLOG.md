# Development log — RAG against the machine

A step-by-step record of what I did and **why**, so I can understand my own work and explain it during the defense.

> How to use: add a new entry at the end of the log every time I finish something (ideally one entry per commit). Keep it short: what, why, and what I learned. A template is at the bottom.

---

## Step 0 — Understanding the project (2026-09-28)

**What**
- Read the subject (`en.subject.pdf`) and made a study guide in Spanish (`guia/`): translation, summary, detailed explanation, exercises, vLLM and PyTorch notes.

**Why**
- The subject says to read everything before coding: each RAG stage (index → retrieve → augment → generate) feeds the next.

**Learned**
- RAG = give the model the right documents at answer time instead of retraining it.
- The grade depends mostly on **retrieval** (recall@k), not on the LLM's wording.
- `vllm-0.10.1` is **not a tool I use**: it is the **corpus** (the data my system answers questions about).
- The LLM is `Qwen/Qwen3-0.6B`, run locally with `transformers` + `torch`. No `llm_sdk` like in call-me-maybe.

---

## Step 1 — Project skeleton (2026-09-28) · commit `3b05268`

**What**
- `uv init` → then fixed the layout it generated (`src/rag/`) so that `src/` itself is the package.
- `src/__init__.py` + `src/__main__.py` → the program runs with `uv run python -m src`.
- `pyproject.toml`: `package = false`, Python `>=3.10`, dependencies `fire`, `pydantic`, `tqdm`; dev group with `flake8`, `mypy`, `pytest`; mypy config.
- `.flake8` and mypy `exclude` → linters skip `.venv` and `data/` (otherwise they would check the ~1,700 vLLM Python files).
- Folder layout required by the subject:
  - `data/raw/vllm-0.10.1/` → corpus
  - `data/processed/` → index
  - `data/datasets/{UnansweredQuestions,AnsweredQuestions}/` → questions
  - `data/output/{search_results,search_results_and_answer}/` → my outputs
- `.gitkeep` files so git keeps the empty folders.
- `.gitignore`: vLLM and moulinette are not tracked (removed from the first commit with `git rm --cached` + `commit --amend` + `push --force`).

**Why**
- The evaluator only runs `uv sync` and `uv run python -m src <command>`, and its scripts expect this exact layout.
- `file_path` in my results is compared **character by character** with `data/raw/vllm-0.10.1/...`.

**Learned**
- `uv sync` makes `.venv` match `pyproject.toml` + `uv.lock` exactly. Never `pip install` by hand; use `uv add` / `uv add --dev`.
- `.gitignore` only affects files git is **not already tracking**.
- `git rm --cached` stops tracking a file without deleting it from disk.

---

## Step 2 — Campus reminders (2026-09-28) · commit `f8b5f6a`

**What**
- Temporary `README.md` with a reminder of what must be downloaded on each machine (moulinette and raw data are not in git).

**Why**
- Those files are ignored by git, so a fresh clone does not have them.

---

## Step 3 — Makefile (2026-09-28) · commit `073270b`

**What**
- Base `Makefile` with `all`, `run`, `debug`, `lint`, `lint-strict`.

**Why**
- The subject requires a Makefile to automate common tasks.

**Pending**
- Rules `install` and `clean` are required by the subject.
- `lint` must use the subject's exact mypy flags.

---

## Step 4 — Data models and errors (2026-09-29) · commit `cba369d`

**What**
- `src/models.py`: the pydantic models from the subject.
  - `MinimalSource` → one source: file + character range.
  - `UnansweredQuestion` → `AnsweredQuestion` (inherits and adds `sources`, `answer`).
  - `RagDataset` → the question files.
  - `MinimalSearchResults` → `MinimalAnswer` (inherits and adds `answer`).
  - `StudentSearchResults` / `StudentSearchResultsAndAnswer` → the output files.
- `src/exceptions.py`:
  - `RagError` → base class for all project errors.
  - `DatasetError(RagError)` → turns a pydantic `ValidationError` into a short `field: message` list.

**Why**
- pydantic validates every JSON I read or write; field names **are** the JSON keys, so they must match the subject exactly (`question`, `retrieved_sources`...).
- The CLI must never crash with a traceback: a single `except RagError` will catch every project error thanks to inheritance.

**Learned**
- `Model.model_validate(dict)` / `Model.model_validate_json(text)` → create validated objects; `model_dump_json()` → back to JSON.
- `e.errors()` gives a list of dicts (`loc`, `msg`, `type`...) → print only what matters.
- `raise NewError(...) from e` → chains exceptions: records that `e` **caused** the new error.
- `list[...]` = `List[...]` (the `typing` one is the old style, before Python 3.9).
- Overriding a `list[Parent]` field with `list[Child]` in a subclass fails in mypy (list is *invariant*); `Sequence` would allow it.
- Keep it simple (YAGNI): don't add features "just in case".

---

## Next steps

- [ ] Docstrings on every model in `models.py`.
- [ ] Complete the Makefile (`install`, `clean`, exact lint flags).
- [ ] Rename `README.md` content / create the real README (English, subject sections).
- [ ] CLI class with Fire exposing the six commands (empty for now).
- [ ] Chunkers (Python + Markdown), BM25/TF-IDF index, `index` command.

---

<!--
## Step N — Title (YYYY-MM-DD) · commit `xxxxxxx`

**What**
-

**Why**
-

**Learned**
-
-->
