# Development log — RAG against the machine

A step-by-step record of what I did and **why**, so I can understand my own work and explain it during the defense.

> How to use: add a new entry at the end of the log every time I finish something (ideally one entry per commit). Keep it short: what, why, and what I learned. A template is at the bottom.

> **Personal reminders** (moved from the old README):
> - Download the moulinette. It's only at home.
> - Download the raw data. It's only at home.
>
> Please, don't be like always :)

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

**Pending** → done in Step 6
- ~~Rules `install` and `clean` are required by the subject.~~
- ~~`lint` must use the subject's exact mypy flags.~~

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

## Step 5 — Devlog (2026-09-29) · commit `79f809c`

**What**
- This file: a step-by-step record of the work.

**Why**
- To understand my own work and be able to explain every decision in the defense.

---

## Step 6 — Makefile complete (2026-09-29) · commit `05e9631`

**What**
- `install` (runs `uv sync`); `all` now depends on it.
- `clean`: removes `__pycache__` (recursively, skipping `.venv`), `.mypy_cache`, `.pytest_cache`.
- `lint` with the subject's exact mypy flags.
- `@` in front of some commands so make does not echo them.

**Learned**
- `all` is only the default rule (the first one); `install` must exist with its own name because the evaluator may call it.
- `find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +` → recursive delete without entering `.venv`.
- `@command` hides the command line, not its output. `.SILENT:` / `make -s` hide all of them.

---

## Step 7 — CLI skeleton (2026-09-29) · commit `f5ef841`

**What**
- `src/commands.py`: class `RagCLI`, one method per subject command (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`), with the subject's option names. Bodies are still empty.
- Extra `--k` option in `evaluate`.

**Learned**
- Fire turns a class into a CLI: method name = command, parameter name = `--option`, parameters without default = required/positional, docstring = `--help`.
- Parameter names **are** the CLI flags → typos break the evaluator scripts (`max_chunk_size`, `student_search_results_path`, `dataset_path`).
- Fire **ignores type hints**: it guesses types from the text (`123` → int, `--k abc` → str).
- Don't load heavy things (index, model) in `__init__`: Fire builds the object for every command.
- Return `None` and print inside the method: Fire prints return values its own way.

---

## Step 8 — Argument validation, entry point and tests (2026-09-29) · commit `7d16c38`

**What**
- Arguments validated with pydantic `validate_call` + `Annotated[..., Field(...)]` (`k > 0`, non-empty query, `FilePath` for input files that must exist).
- `CliCommand.validated`: my own decorator. It applies `validate_call` with a shared config (`coerce_numbers_to_str`, `str_strip_whitespace`), gives Fire a clean signature (`__signature__` without `Annotated`) so `--help` shows `int`/`str`/`Path`, and removes `raw_function` from the help.
- Docstrings on the module, class and every command (Fire shows them in `--help`).
- `ArgError(RagError)`: shows `'value': reason` for each invalid argument.
- `src/__main__.py`: `fire.Fire(RagCLI)` inside two `try` levels.
  - Inner: `ValidationError` → `raise ArgError(e) from e`.
  - Outer: `RagError` → message to stderr, exit 1. `KeyboardInterrupt` → exit 130. `Exception` (last) → `Unexpected error: Type: msg`, exit 1.
- Tests (`make test`, 52 passing):
  - `tests/test_cli_parsing.py`: valid, invalid and wrong-usage command lines, run with `fire.Fire(RagCLI, command=[...])`.
  - `tests/test_exceptions.py`: message format and inheritance of `RagError`, `DatasetError`, `ArgError`.
- `pyproject.toml`: pytest config (`testpaths = ["tests"]` so it skips vLLM's own tests; `pythonpath = ["."]` to `import src`).

**Why**
- The subject tests the CLI with degenerate inputs (empty query, `k=0`, missing files...) and it must never show a traceback.
- Declarative validation: the constraints live in the method signature, no `if` checks.

**Learned**
- `FilePath` + `Field(min_length=1)` crashes with a `TypeError` (a `Path` has no length). mypy does not catch it → **run the code**, don't trust only the linters.
- pydantic does **not** validate default values.
- `Field(...)` as a default value breaks Fire (it passes the `Field` object itself).
- In validation errors from Fire, `loc` is the **position** of the argument, not its name → show the input value instead.
- `BaseException` is the root; `Exception` is the root of real errors. Never catch `BaseException`: it includes `SystemExit` (Fire's usage errors) and `KeyboardInterrupt`.
- `except` order: most specific first, `Exception` last.
- A `raise` inside an `except` is not caught by the sibling `except` clauses of the same `try` → nested `try` needed.
- Exit codes: 0 ok, 1 error, 2 usage (Fire), 130 Ctrl+C. Errors go to `stderr`.
- `@pytest.mark.parametrize` → one test method, many cases.

---

## Step 9 — Corpus reader (2026-09-30) · commit `316569c`

**What**
- `src/reader.py`: `CorpusReader(path_to_read).read() -> CorpusReadResult` (`documents` + `skipped`, property `found`).
  - Class attributes `EXTENSIONS` (`frozenset` of `.py`, `.md`, `.txt`) and `PROJECT_ROOT` (folder containing `src`).
  - Lists with `rglob` + comprehension (only allowed extensions, only files, sorted), reads with `read_text`, stores the path **relative to the project root** (`data/raw/vllm-0.10.1/...`).
  - `CorpusError` when the folder is missing, outside the project, has no valid files, or none is readable. Unreadable files go to `skipped`.
- Models `Document`, `CorpusReadResult`. **Decision: every pydantic model lives in `models.py`.**
- `index` command wired to the reader (summary: files read, characters, files per extension). `--raw_dir` option (default `data/raw`) — paths are never hard-coded.
- Pending code now raises `NotImplementedError("<name> is not implemented yet")` (my convention: `pass` = implemented by subclasses).
- Tests for reader, models, main, decorator.

**Learned**
- The moulinette compares `file_path` **verbatim**: with absolute paths recall is 0 % (checked).
- `Path(...)` never checks the disk; `is_dir()`/`is_file()` do. `rglob` on a missing folder returns `[]` silently.
- `relative_to` needs both paths absolute → `resolve()` the folder once in `__init__`.
- Return results, don't store them in the object (no half-built objects).
- `--raw_dir ../` was accepted by accident → folder must be `is_relative_to(PROJECT_ROOT)`, checked before listing.
- `\r\n` investigated: no change needed (see Reader follow-ups).

---

## Step 10 — Chunker size guard (2026-09-30) · commit `451dcbf`

**What**
- `BaseChunker.__init__`: `TypeError` if `max_chunk_size` is not a real `int` (a `bool` is rejected too), `ValueError` if < 1 (0 would loop forever).

**Learned**
- `bool` is a subclass of `int`. Check the type first, then the value. `TypeError` = wrong type, `ValueError` = wrong value.

---

## Step 11 — Chunker, tokenizer, BM25 index, stricter CLI (2026-09-30) · commit `d9a7300`

**What**
- `src/chunker.py`: `BaseChunker` (ABC, template method: `chunk_documents` loops with tqdm, subclasses implement `chunk_document`) and `FixedSizeChunker`.
  - Cuts at `\n\n` → `\n` → `' '` → hard cut, only past half the window (no tiny chunks), never inside the last piece. Blank pieces are skipped. Offsets are exclusive, text is never modified.
  - Model `Chunk(MinimalSource)` = source + text. 12,134 chunks at 2000 chars.
- `src/tokenizer_bm.py`: `Tokenizer.tokenize` = lowercase + `[a-z0-9]+` (compiled once).
- `src/bm25_index.py`: `BM25Index` with `K1 = 1.5`, `B = 0.75`.
  - `build(chunks)` (classmethod): postings `term → {chunk id: tf}` + lengths; `IndexBuildError` if there are no tokens at all.
  - `search(query, k) -> list[SearchHit]`: distinct query terms, idf × saturated tf with length normalisation, top-k with `heapq.nsmallest`, **ties broken by chunk id**.
- `src/argv_guard.py`: `ArgvGuard` checks the command line **before** Fire runs it: unknown options, too many arguments, option without value → `UsageError` (exit 2); any `--help` becomes `<command> -- --help`.
- `ArgError` now names the option: `--k '0': Input should be greater than 0`; blank text → `must not be empty or blank`. Raised by the decorator itself; `main` has a single `try`.
- Tests: chunker invariants (random texts + real corpus), BM25 against an independent naive reference, argv guard. Tests checked by breaking the code on purpose (mutations).

**Learned**
- Fire ignores type hints and **runs the command before** complaining about unknown options, extra arguments or a late `--help`; an option without value becomes `True`. Hence ArgvGuard.
- BM25: tf = occurrences in a chunk, df = chunks containing the term, idf = rarity, saturation (`K1`), length normalisation (`B`).
- The order of a `set` of strings changes **between runs** (hash randomization) → without a tie-break the same query could rank differently each time.
- Classmethods as alternative constructors: `build` and `load` both end in `cls(...)`, so an index is always complete.
- Run the code: `unit="chunk"` typo (`ile=`) and `encoding="uft-8"` passed mypy but crashed.

---

## Step 12 — Save the index (2026-09-30) · commit `d7428b0`

**What**
- `BM25Index.save(index_dir)`: `IndexData` (chunks, postings, lengths) as JSON in `index_dir/index.json`; creates the folder; `PathError` on any `OSError`.
- `index` builds and saves the index. New option `--index_dir` (default `data/processed`).

---

## Step 13 — Load the index and `search` (2026-09-30) · commit `cd46cff`

**What**
- `BM25Index.load(index_dir)` (classmethod): reads `index_dir / FILE_NAME`.
  - Missing file → `IndexNotFoundError` ("No index found in '...'. Run 'index' first.").
  - Other `OSError` → `PathError`. Invalid content → `DatasetError`.
- `search <query> --k --index_dir`: prints a **`StudentSearchResults` JSON** with one entry (subject VI.5 + VI.7.2: single-query commands behave like the dataset ones).
- `RagCLI._search_question(bm25, question, k) -> MinimalSearchResults`: search + convert each hit into an explicit `MinimalSource` (no text, no score). **Reused by `search_dataset`.**
- Docstrings for the BM25 module and `index`.
- Tests: save/load round trip, missing/corrupt index, `search` output format. Tests never write into `data/processed` (they use `--index_dir` with `tmp_path`). 630 tests passing.

**Numbers**
- Read + chunk + build: ~0.7 s. Load from JSON: ~0.2 s (32 MB). One query: 4-14 ms.
- Dataset question "What HTTP endpoint is used to dynamically load a LoRA adapter?" → top-1 hit overlaps the ground truth (IoU ≈ 0.45).

---

## Step 14 — `search_dataset` and first moulinette run (2026-09-30 / 10-01) · commits `d938f2c`, `49010c9`, `ffa564a`

**What**
- `search_dataset --dataset_path --k --save_directory --index_dir`: loads the index once, reads the dataset, searches every question (tqdm) with `_search_question` **keeping the dataset `question_id`**, writes a `StudentSearchResults` JSON to `save_directory/<same file name as the dataset>`.
- `RagCLI._open_file_save(path, model) -> model`: generic reader for every JSON input (`TypeVar M bound=BaseModel`). Read error / not UTF-8 → `PathError`; invalid content → `DatasetError`.
- `dataset_path` is **required** (no default dataset file).
- `ArgvGuard` extended: a command name given as a value (`index -m 500 search`), text after `-` (chaining), unknown flags after `--`. `ArgError.describe` says when a value was taken as positional (`'patata' (positional, taken as --max_chunk_size)`).
- `PathError` also accepts `UnicodeDecodeError`. Docstrings across `reader`, `chunker`, `tokenizer_bm`, `models`, `exceptions`. `IndexData` checks its own consistency (`model_validator`).
- 704 tests passing at this point.

**Baseline recall** (FixedSizeChunker 2000, BM25 own, `K1 = 1.5`, `B = 0.75`, tokenizer `[a-z0-9]+`):

| Dataset | @1 | @3 | **@5** | @10 | Target @5 |
|---|---|---|---|---|---|
| docs | 61.0 % | 78.0 % | **85.0 %** | 88.0 % | ≥ 80 % ✅ |
| code | 31.3 % | 53.5 % | **59.6 %** | 67.7 % | ≥ 50 % ✅ |

- Timings: `index` 0.95 s (limit 5 min), `search_dataset` ~1.5 s per dataset (limit 90 s / 200 questions).
- Edge cases checked: not UTF-8, broken JSON, empty dataset, folder instead of file, `save_directory` is a file, huge k, no index, AnsweredQuestions dataset as input (ids kept).

**Learned**
- `dataset.rag_questions` is already a list of `UnansweredQuestion` objects (no dicts); an `AnsweredQuestion` is one too (inheritance).
- Creating `UnansweredQuestion(question=...)` generates a **new** id → the moulinette could not match anything (recall 0).
- `return` inside a `try` works as usual; the `try` only catches what fails on that line.

---

## Step 15 — `evaluate`: own recall@k (2026-10-01) · commits `edc43e2`, `7556514`

**What**
- `src/evaluator.py`: `Evaluator` receives the **already loaded** models (no paths, no file reading).
  - `_validate_dataset` → `dict[question_id, correct sources]`. Empty dataset, a question that is not an `AnsweredQuestion` or has no sources → `DatasetEvaluationError` (so recall never divides by 0).
  - `_validate_results` → `dict[question_id, retrieved sources]`. Repeated id, more sources than `k`, impossible range, source longer than `MAX_CNTX_LEN` (2000, like the moulinette) → `ResultsEvaluationError`. Empty results are allowed.
  - `_iou(a, b)`: intersection / union of two character ranges. `_is_found`: same `file_path` **and** IoU ≥ `MIN_IOU` (0.05). `_question_recall`: found / number of correct sources.
  - `evaluate(results, dataset, k)`: `missing` (in the dataset, not in the results → recall 0) and `unknown` (in the results, not in the dataset → ignored); cutoffs `CUTOFFS = (1, 3, 5, 10)` that are ≤ `k` and ≤ `results.k`; recall@cutoff = sum / number of dataset questions. Returns `EvaluationInform`.
- `DatasetEvaluationError` / `ResultsEvaluationError` (subclasses of `EvaluationError`, which keeps the bare message in `problem`): the `evaluate` command catches each one and re-raises it with **its** file path.
- **Checked: identical numbers to the moulinette** on docs and code (table above).

**Learned**
- IoU: a 2000-char chunk containing a 40-char answer has IoU 0.02 → not found. Smaller chunks can score better.
- A comprehension cannot `raise` → use a loop when building and validating at the same time; `isinstance` in the loop also narrows the type for mypy (`all(...)` + comprehension does not).
- `isinstance(q, UnansweredQuestion)` is True for answered questions too (inheritance): check `not isinstance(q, AnsweredQuestion)`.
- `or` short-circuits: `not isinstance(q, AnsweredQuestion) or not q.sources` never reads `sources` of an unanswered question.
- A syntax error (empty method body) stops mypy entirely: read the last line of its output.
- Imports inside the package need the dot (`from .models`): the program runs from the project root, not from `src/`.
- Exceptions propagate on their own; catch in an intermediate function only to add information (here: the file path).

---

## Step 16 — `evaluate` finished, test data, stricter CLI (2026-10-01) · commit `64a048f`

**What**
- `evaluate` prints the report: `Student data is valid: True`, recall per cutoff, unknown / missing questions (at most `MAX_IDS_SHOWN = 5`, then `... and N more`), `Found within k=…: X/Y questions`.
- `evaluate --show_failed`: lists every question not fully found within the first k results, with its text. **Same list as the moulinette's `list_valid_questions`** (checked: 15 docs, 40 code, 3 mini hard).
- `Evaluator.evaluate` returns `EvaluationInform` (read-only properties `recall`, `missing`, `unknown`, `failed`). `DatasetEvaluationError` / `ResultsEvaluationError`: the command re-raises each one with the path of **its** file.
- `ArgvGuard`: a first word that is not an option nor a public command → `Unknown command` (Fire would otherwise run private methods like `_print_ids` and print class attributes like `SEPARATOR`).
- Test data in `tests/data`: a mini corpus (`raw/mini_vllm`, 10 files) and two datasets with exact ranges: `dataset_mini.json` (11 easy) and `dataset_mini_hard.json` (13 hard, each with a `difficulty`: paraphrase, identifier, small-answer, distractor). Real search results of docs/code for a regression test.
- `make test_rag` (index → search_dataset → evaluate on the mini corpus); options `TEST_DATASET`, `TEST_K`, `TEST_CHUNK`. Generated files in `tests/data/processed` and `tests/data/output` (ignored by git, removed by `make clean`).
- Docstrings on every module, class and function of `src/` and `tests/` (checked with an AST scan).
- README: "Commands (copy & paste)" for every command and the moulinette (`--threshold`, `list_valid_questions`).
- 808 tests passing.

**Learned**
- Recall@k uses the first k results: an answer in 4th position counts for @5 and @10, not for @1 and @3.
- Iterating a dict gives its keys; `.items()` gives pairs. Format spec is `{value:.3f}`, not `:f3`.
- stdout is buffered when it is not a terminal and stderr is not: piped output can look out of order (call `sys.stdout.flush()` before writing errors if needed).
- Fire only hides `_private` methods from the help; named explicitly, it runs them.

---

## Step 17 — `PyChunker`, dotted terms, path experiment (2026-10-02) · commit `19b43eb`

**What**
- `PyChunker(FixedSizeChunker)`: only `SEPARATORS` and `CUT_BEFORE`. Prefers to cut right before a definition: `\n@`, `\nclass `, `\ndef `, `\nasync def `, then the indented ones (methods), then `\n\n`, `\n`, space. `CUT_BEFORE` separators cut after their leading newline, so the definition starts the next chunk.
- One single `_find_cut` in `FixedSizeChunker` (`CUT_BEFORE` empty there): checked that both chunkers give identical chunks on the whole corpus before/after moving it.
- **Not used yet**: `index` still chunks every file with `FixedSizeChunker`.
- Tokenizer second pass `COMPLEX_PATTERN` (`[a-z0-9_]+(?:\.[a-z0-9_]+)+`): dotted names are also kept whole (`llm.generate` → `llm`, `generate`, `llm.generate`). Tests updated + 22 new ones (830 passing).
- Experiment (commented in `BM25Index.build`): also tokenize `chunk.file_path`. 35 of 99 code questions contain words that are only in the path of their file (`keye`, `sampler`, `fused_moe`...).

**Results** (full tables in README → Performance)
- `PyChunker` vs `FixedSizeChunker` on code: no clear gain. At 2000 chars recall@1 +5 points but recall@5 −5 (gains 1 question, loses 6); at 500 recall@5 +2; at 1000 about the same.
- Why: 4 of the 6 lost questions still retrieve the right file but another chunk of it ranks first; ground-truth ranges are arbitrary ~880-char windows and IoU ≥ 0.05 is a low bar, so the position of the cuts barely matters. What decides a hit is the words → next levers: path and identifiers in the indexed text.
- Kept anyway: the subject requires a Python-specific strategy.

**Learned**
- Measure every change on its own (one at a time, rebuild the index, `evaluate`). With ~100 questions, 1 question ≈ 1 %.
- A cleaner structure is not always better recall.

---

## Step 18 — Two chunking strategies, tokenizer and query tuning (2026-10-02 / 10-03) · commits `18ed1fc`, `c9e1b4d`, `5233008` (+ uncommitted)

**What**
- Tokenizer: snake_case pass (`max_model_len` kept whole). The dotted pass was **removed**: it barely helped (~1 code question) and added terms to index.
- File path tokens enabled in `BM25Index.build`; `K1 = 1.9`, `B = 0.5`.
- `MdChunker(FixedSizeChunker)` for `.md` **and `.txt`** (the subject's "Markdown / text" strategy): first separator `\n```\n`, so a code block stays whole in the chunk it closes. Heading separators were measured and dropped (−4 docs points). `index` now uses exactly two strategies: `PyChunker` for `.py`, `MdChunker` for the rest.
- `PyChunker._find_cut`: decorators kept with their definition (see "Solved" below).
- `Tokenizer.STOPWORDS` + `tokenize_query`: question words and common words are dropped from the **query only**; `search` uses it.
- Tests: `TestPyChunker` (+ real corpus), `TestMdChunker`, `TestQueryStopwords`; dotted tests removed (1031 passing).
- Makefile: `test_rag_base` saved to `data/output/search_results` instead of `.../UnansweredQuestions` (the folder was never created) — fixed.

**Results** (moulinette, chunk 2000, k = 10, recall@5; full tables in README → Performance)
- Baseline 2026-10-01: docs 0.85, code 0.596.
- Now: **docs 0.87, code 0.758**. Index ~2 s, each public dataset searched in < 1 s.
- Overlap between chunks (experiment, not implemented): hurts docs (neighbour chunks take two of the five slots); on `.py` only +1–4 code questions, not growing with the overlap → noise.

**Learned**
- idf is not enough against question words in a code corpus: `why`/`how`/`what` are rare there, so they weigh more than `lora`.
- Stop tuning when the gains are the size of the noise (1 question ≈ 1 %).

---

## Step 19 — Generation: `Generator` and first prompt experiments (2026-10-03) · in progress

**What**
- `torch` CPU-only (own uv index in `pyproject.toml`) + `transformers`.
- `Generator`: loads Qwen3-0.6B once (`float32`), `_build_context` (chunks with a header and their path until `MAX_CONTEXT_TOKENS`, counted with Qwen's tokenizer), `_build_messages` (system + user, question last, reminder after it), `generate` (chat template without thinking, greedy, only the new tokens decoded).
- `answer` prints the raw answer (JSON output and the no-hits case pending).

**Timings (CPU, 8 threads)**
- Model load ~5.6 s. Prompt of ~2000 tokens: prefill **29 s in bfloat16** (the default: Qwen is stored in bf16 and this CPU has no native bf16) vs **10 s in float32**; ~150 ms per generated token. → `dtype=torch.float32`.
- No time limit on generation in the subject (the 90 s are for retrieval), but `answer_dataset` on 100 questions will take ~20–40 min.

**Prompt experiment: "What CUDA architectures are supported for Marlin MOE kernels in vLLM?"** (expected `8.0;8.7;9.0+PTX`, in `CMakeLists.txt`; retrieval OK: the right chunk is 2nd)

| Variant | Answer |
|---|---|
| 6 chunks, 4-rule prompt | ❌ "…are specified in the context…" |
| Oracle: only the right chunk | ✅ |
| Reversed order (best chunk last) | ❌ |
| 3 chunks, 4-rule prompt | ❌ invents `MARLIN_ARCHS` |
| 6 or 3 chunks, short prompt "Quote the exact names and values" | ✅ `8.0;8.7;9.0+PTX` |

Same battery, old vs new prompt: bitsandbytes min version ✅/✅; `--turbo-mode` (does not exist) refuses ✅/✅; LoRA loading ✅/✅ (new one more direct); Mixtral quantization methods ❌/❌ (new one copies an exact list, but from the wrong chunk); "capital of France" ❌/❌ (answers "Paris" from its own knowledge). → Short prompt adopted.

**Learned**
- The oracle test (only the right chunk) separates retrieval failures from generation failures.
- With a 0.6B model, fewer and positive instructions work better than a list of rules. Some failures (own knowledge, picking the wrong list) are model limits.

**Off-topic rule (added to `REMINDER`, not measured yet)**
- Idea: "If the question is not about vLLM, LLMs, model serving or programming, say: <the fixed refusal>". Judging the topic of the question is easier for a small model than checking whether the context contains the answer, so it may fix "Paris".
- Risks to check: valid questions that do not name vLLM ("How do I load a LoRA adapter?") refused; a longer reminder weakens "exact values" (Marlin). Re-run the whole battery, not only "Paris".
- Plan B (no prompt): skip the model when the best BM25 score is very low (the threshold is hard to choose).
- **Measured (2026-10-03):** "Paris" is now refused ✅, `--turbo-mode` refused ✅, bitsandbytes ✅, but **Marlin and LoRA (valid, correct before) are refused ❌**. The risk happened. To decide: shorter wording, drop the rule, or plan B.

**Answer battery (`make test_answer`)**
- 10 questions in `tests/answer_battery/`, one file each (`QUESTION:` / `EXPECTED:` lines): 4 docs and 4 code questions from the public datasets (LoRA reworded so it does not name vLLM), 1 off-topic ("capital of France") and 1 trap (`--turbo-mode`, does not exist). The rule runs `answer` on each one and prints the expected answer after it. Extra options: `ANSWER_ARGS="--k 5 --dtype bfloat16"`. ~4 min for the 10.
- Not in `tests/data/` because that folder is in `.gitignore`.

**Decoding: greedy vs sampling (Qwen docs read on 2026-10-03)**
- The Qwen3-0.6B model card recommends sampling: thinking mode `temperature=0.6, top_p=0.95, top_k=20` and *"DO NOT use greedy decoding"*; non-thinking mode (ours) `temperature=0.7, top_p=0.8, top_k=20, min_p=0`. The model's `generation_config.json` has `do_sample: true` with the thinking-mode values.
- The subject says nothing about decoding (no greedy, sampling, temperature or seed). The only "reproducible" (VI.7.1) is about the retrieval pipeline layout. So both are allowed.
- Battery with the current prompt (short system prompt + reminder with the off-topic rule):

| # | Question | Greedy (`do_sample=False`) | Sampling (`do_sample=True`, defaults of `generation_config.json`) | **Greedy, no off-topic rule** |
|---|---|---|---|---|
| 01 | bitsandbytes version | ✅ 0.46.1 | ✅ 0.46.1 | ✅ 0.46.1 |
| 02 | Marlin MOE architectures | ❌ refuses | ❌ refuses | ✅ `8.0;8.7;9.0+PTX` |
| 03 | load a LoRA adapter | ❌ refuses | ❌ refuses | ✅ endpoint, fields and example |
| 04 | security issues | 🟡 "privately, GitHub security advisory" | ❌ refuses | 🟡 same |
| 05 | Mixtral quantization | ❌ refuses | ❌ refuses | ❌ exact list from another chunk |
| 06 | `pad_to` default | ❌ "1" (invented) | ❌ "1" (invented) | ❌ "1024" (invented) |
| 07 | HybridKV block size | ✅ (plus an extra sentence) | ❌ refuses | ❌ relation reversed |
| 08 | `use_bitsandbytes_4bit` default | ❌ refuses | ❌ refuses | ✅ `False` |
| 09 | capital of France | ✅ refuses | ✅ refuses | ❌ "Paris" |
| 10 | `--turbo-mode` | ✅ refuses | ✅ refuses | ✅ refuses |

- Valid questions refused: **4 greedy → 6 sampling**. Sampling makes the same problem worse: the model already hesitates between answering and refusing, and randomness adds to it.
- Caveats: one run only (sampling changes between runs; fix `torch.manual_seed(0)` to compare), and the sampling run used the thinking-mode defaults, not the recommended non-thinking values.
- **Decision: keep greedy.** Better here, and reproducible, so the battery compares prompts fairly.
- The real problem is the prompt: 4 of 8 valid questions refused because of the off-topic rule. Next: drop that sentence from `REMINDER` (without it Marlin and LoRA were right, only "Paris" failed), then try a few-shot refusal example instead of the rule.

**Off-topic rule dropped (greedy)** — `REMINDER` is now only *"Answer with the exact values from the context."*; `SYSTEM_PROMPT` unchanged (last column above).
- Valid questions: **2 right + 1 partial → 4 right + 1 partial** of 8 (02 Marlin, 03 LoRA, 08 `False` fixed). No valid question refused any more.
- Lost: 09 "Paris" again (expected without the rule) and **07 now reverses the relation**: a wrong answer that looks right, worse than a refusal.
- Why the remaining failures happen (right chunk's rank in the BM25 results vs chunks that fit in `MAX_CONTEXT_TOKENS = 2500`):

| # | Right chunk rank | Chunks in context | Kind of failure |
|---|---|---|---|
| 04 security issues | 2 | 7 | generation (partial answer) |
| 05 Mixtral | **7** | **6** | **retrieval / budget**: the right chunk never reaches the model, so it copies a list from another chunk |
| 06 `pad_to` | 1 | 6 | generation: the answer is a constant name, the model invents a number |
| 07 HybridKV | 1 | 5 | generation: reads the right chunk and reverses it |

- Next: a few-shot refusal example for "Paris" (teach by example instead of a rule); then try a larger `MAX_CONTEXT_TOKENS` for 05 (costs prompt time: ~2,000 tokens ≈ 10 s).

**`MAX_CONTEXT_TOKENS` 2500 → 3500** (same prompt, greedy) — worse, reverted:

| # | 2500 | 3500 |
|---|---|---|
| 01 bitsandbytes | ✅ | ✅ |
| 02 Marlin | ✅ | 🟡 right values, wrong variable name (`CUDA_ARCHS`) |
| 03 LoRA | ✅ | ❌ quotes `lora_name="adapter"` from a test, no endpoint |
| 04 security | 🟡 | ❌ copies the list of steps of the security team |
| 05 Mixtral | ❌ list from another chunk | ❌ 2 of 6 right (`gptq_marlin`, `awq_marlin`) plus a wrong one (`ipex`) |
| 06 `pad_to` | ❌ "1024" | ❌ "1" |
| 07 HybridKV | ❌ reversed | ❌ reversed |
| 08 `use_bitsandbytes_4bit` | ✅ | ✅ |
| 09 France | ❌ | ❌ |
| 10 turbo-mode | ✅ | ✅ |

- Valid questions: 4 right + 1 partial → **2 right + 1 partial**. More chunks = more distractors: the model copies values from the wrong chunk (03 from a test file, 04 a nearby list). Also ~5 s slower per answer.
- 05 gets part of the right list (its chunk now fits) but mixes it with others: more context does not fix it.
- Side effect seen in 02, 03 and 07: the model repeats the reminder ("The exact values are: …"). The wording of `REMINDER` leaks into the answers.
- **Decision: keep 2500.**

**Few-shot examples, then a new `REMINDER`** (greedy, `MAX_CONTEXT_TOKENS = 2500`) — **best prompt so far**
- Few-shot: two invented example turns before the real question (`Generator._build_messages` / `_user_message`): one answered by quoting the context (`max_retries = 3` in a fake `ServerConfig`) and one unrelated question refused ("Who wrote Don Quixote?" → the fixed `REFUSAL`). Teaching by example instead of the off-topic rule. ~200 extra prompt tokens (< 1 s).
- Then `REMINDER` changed from *"Answer with the exact values from the context."* (the model copied it into its answers: "The exact values are: …") to *"Answer the question using the context above."* The "exact values" instruction stays in `SYSTEM_PROMPT` and is shown by the example.

| # | No off-topic rule | + few-shot | **+ new `REMINDER`** |
|---|---|---|---|
| 01 bitsandbytes | ✅ | ✅ | ✅ |
| 02 Marlin | ✅ | ⚪ refuses | ✅ `8.0;8.7;9.0+PTX` (calls the variable `CUDA_ARCHS`, minor) |
| 03 LoRA | ✅ | ❌ values from a test file | ✅ endpoint + `curl` example |
| 04 security | 🟡 | 🟡 | 🟡 "follow the security policy" |
| 05 Mixtral | ❌ list from another chunk | ⚪ refuses | ⚪ refuses (honest: the right chunk does not fit in the context) |
| 06 `pad_to` | ❌ invents "1024" | ✅ `DEFAULT_VOCAB_PADDING_SIZE` | ✅ |
| 07 HybridKV | ❌ reversed | 🟡 incomplete | 🟡 quotes the source (see below) |
| 08 `use_bitsandbytes_4bit` | ✅ | ✅ | ✅ |
| 09 France | ❌ "Paris" | ✅ refuses | ✅ refuses |
| 10 turbo-mode | ✅ | ✅ | ✅ |
| **Score** (✅ 2, 🟡 1, ⚪ 0, ❌ −1) | 7 | 11 | **16** |
| **Hallucinations** (wrong and confident) | 4 | 1 | **0** |

- Few-shot alone: same number of right answers, hallucinations 4 → 1 ("Paris" refused, `DEFAULT_VOCAB_PADDING_SIZE` quoted like the example quotes a name). The new reminder then recovered Marlin and LoRA: the old wording was leaking into the answers.
- Many general-knowledge questions tried by hand: all refused. Next to try: off-corpus questions that *sound technical* (Python GIL, CUDA unified memory, installing PyTorch with conda), where the model knows the answer and the context looks relevant.
- **07 is a finding about the source, not a hallucination.** `vllm/v1/core/kv_cache_coordinator.py:297` checks `other_block_size % full_attention_block_size == 0` (other divisible by full, as the dataset says), but its assertion message says *"assumes the block_size of full attention layers is divisible by other layers"*: the opposite. The model quotes that message. Grounded does not always mean correct when the source contradicts itself.
- Remaining: 04 (partial) and 05 (retrieval: the right chunk ranks 7th, only 6 fit).

**`MAX_CONTEXT_TOKENS` 2500 → 2800 with the best prompt** — chosen as the smallest budget where the Mixtral chunk (rank 7) fits; most questions get one more chunk. Worse, reverted:
- Score **16 → 13**, still no confident hallucination. Changes: 02 better (no wrong variable name); **03 broken** (answers with a heading: "`Load LoRA Adapter`"); **05 still refused although its chunk is now in the context**. The rest is unchanged.
- Second time a bigger context loses (3500 earlier): with this 0.6B model, one more chunk distracts more than it helps, and having the right chunk is not enough (05). **Keep 2500**; the context budget is closed.

**Thinking mode (`enable_thinking=True`) — tried with a temporary `--think` flag, discarded**
- Set-up: a copy of the good `generator.py` kept aside, a temporary `--think` flag on `answer`. In thinking mode: sampling with the model card's values (`temperature=0.6, top_p=0.95, top_k=20`, the card says not to use greedy here), `torch.manual_seed(0)`, up to 2048 new tokens, only the text after `</think>` (token id 151668) returned.
- RAM: only the KV cache grows (~0.23 MB per token in float32, so ~0.3–0.6 GB more for 1,000–2,500 reasoning tokens), below the loading peak. The real cost is time (~150 ms per token).

| Test | Reasoning | Answer | Time |
|---|---|---|---|
| 05 Mixtral, `MAX_CONTEXT_TOKENS = 2500` (right chunk not in the context) | 791 tokens | refusal (correct: the data is missing) | 150 s |
| 05 Mixtral, `MAX_CONTEXT_TOKENS = 2800` (right chunk in the context) | 249 tokens | ❌ `gptq_marlin_24` and `gptq_marlin`: 1 of 6 right plus one from another chunk's list | 65 s |

- With the right chunk in the context and reasoning, the model still picks the wrong list among similar chunks, and turns an honest refusal into a confident wrong answer. 3–7× slower (20 s → 65–150 s per answer; ~100 questions would take hours).
- **Decision: discarded.** Back to the good `generator.py` (greedy, no thinking, `MAX_CONTEXT_TOKENS = 2500`); flag and copy removed.

**Context budget sweep with the best prompt** (greedy, few-shot, new `REMINDER`)

Whole battery (score: ✅ 2, 🟡 1, ⚪ 0, ❌ −1):

| `MAX_CONTEXT_TOKENS` | Score | Confident wrong answers | Changes against 2500 | Time per answer |
|---|---|---|---|---|
| 1500 | 12–14 | 0 | 03 LoRA ❌ (answers with a heading), 04 security ✅, 07 quotes another condition | |
| 2000 | 15 | 1 (07 reversed again) | 03 ⚪ refuses, 04 ✅ | ~15 s (battery 159 s) |
| **2500** | **16** | **0** | — | ~20 s |
| 2800 | 13 | 0 | 03 ❌ heading, 05 still refused with its chunk in the context | |
| 3500 | (older prompt) | | more chunks copied from the wrong place | |

Questions 03 and 04 only, with the budget changed in memory (`Generator.MAX_CONTEXT_TOKENS = N` in a `python -c` loop, without editing the file):

| Budget | Chunks in context (03 / 04) | 03 LoRA | 04 security |
|---|---|---|---|
| 2000 | 5 / 6 | ⚪ refuses | ✅ "privately, vulnerability submission form on GitHub" |
| 2100 | 5 / 6 | ⚪ refuses | ✅ |
| **2200** | **6 / 6** | ✅ endpoint + `curl` example | ✅ |
| 2300 | 6 / 7 | ✅ | 🟡 "follow the security policy" |
| 2400 | 6 / 7 | ✅ | 🟡 |
| 2500 | 6 / 7 | ✅ | 🟡 |

- The answers only change when the **number of chunks** changes (03 at 5 → 6 chunks, 04 at 6 → 7): the budget matters only through how many chunks fit.
- **2200 is the only budget where both are right.**
- The cause is not logical, it is sensitivity: for 03 both `docs/features/lora.md` chunks are already in the context with 5 chunks; the 6th one that makes it answer is a test file (`test_lora_adapters.py`). For 04 the 7th chunk (`docs/usage/security.md`) makes the answer vaguer. A 0.6B model is fragile to the exact mix of chunks, so 2200 may just be good for these two.
- Confirmed with the whole battery at **2200**: score **17** (best so far), still **0 invented answers**. 01–04, 06, 08 ✅; 05 refused (honest); 07 quotes vLLM's own misleading assertion message, as with 2500; 09 and 10 refused. Battery 190 s (~19 s per answer).
- **Decision: `MAX_CONTEXT_TOKENS = 2200`.** Final prompt: short `SYSTEM_PROMPT` with the fixed `REFUSAL`, two few-shot examples, `REMINDER` *"Answer the question using the context above."*, greedy, no thinking, `MAX_NEW_TOKENS = 150`. Generation tuning closed: the remaining failures (05, 07) are model or source limits.

**Full docs dataset with the final prompt** (`make test_answer_dataset`, 100 questions of `dataset_docs_public.json`, `MAX_CONTEXT_TOKENS = 2200`, greedy). Output saved by hand in `answer.log` (the terminal copy lost some spaces at line breaks, e.g. `andWARP_SIZE`; questions were matched to the dataset ignoring whitespace). Each answer judged by hand against the reference answer, and crossed with retrieval (is the right chunk among the chunks that fit in the context?).

| Result | Count | Right chunk in the context | Right chunk not in the context |
|---|---|---|---|
| ✅ correct | **50** | 48 | 2 (found in another chunk) |
| 🟡 partial / vague / correct but with a contradictory tail | 19 | 17 | 2 |
| ⚪ refusal | 20 | 16 | 4 (honest: the data was missing) |
| ❌ wrong answer | 11 | 8 | 3 |

- Score (✅ 2, 🟡 1, ⚪ 0, ❌ −1): **108 / 200**. Correct or partial: **69 %**. Wrong and confident: **11 %**.
- Retrieval vs generation: the right chunk reached the model in **89 / 100** questions, so most of the loss is in generation. With the chunk in the context: 48 ✅, 17 🟡, 16 ⚪, 8 ❌.
- Time: **2,134 s (35.6 min)**, mean **21.3 s**, median 19.9 s, min 13.8 s, max 57.8 s. Refusals 16.3 s on average, answers 22.6 s. 6 answers above 30 s, all long lists or code blocks (13, 54, 55, 56, 70, 96).

Patterns found:
1. **Few-shot leak (4 answers: 18, 30, 46, 97).** The example answer ends with `(ServerConfig)` and the model copies it: "`--runner pooling` (ServerConfig)", and in 18 the whole example ("`max_retries = 3` (ServerConfig)"). Fix to try: an example answer without the parenthetical source, or with a less copyable one.
2. **Answer plus refusal (6 answers: 24, 44, 60, 80, 82, 86).** A correct answer followed by "The context does not contain the answer.", which contradicts it. Fix to try in code: if the answer has more than the refusal sentence, remove that sentence at the end.
3. **Over-refusal (16 refusals with the right chunk in the context).** The main remaining loss: questions about where to find something ("Where can I find…", 6, 10, 38, 91) and broad questions (4, 9, 26). The price of 0 hallucinations on the battery.
4. **Wrong answers (11):** answering from a nearby but wrong chunk (39 logging variables for multimodal LoRAs, 49 quantization lists, 68 requirements), or with the data missing (3 `encode` instead of `embed`, 22 `FP4PerGroupSpec` instead of `quantization='quark'`).
5. Narrow references: some answers judged 🟡 are fuller than the reference (27 extra structured-output backends, 90 more OpenAI APIs, 52 another audio model).

`answer_dataset` estimate with these times: ~36 min per 100 questions (the model is loaded once there, ~5 s less per question: ~30 min).

**Regression after fixing patterns 1 and 2** (`EXAMPLE_ANSWER` without the trailing `(ServerConfig)`; `Generator._clean_answer` drops the refusal sentence from an answer that has content). Same 100 questions, judged by hand against the first run. Questions 1–76 in `data/output/answer_logs/dataset_docs_public.log`; the run then hung on question 77 (see below) and 77–100 were run again with `ANSWER_START=77` (`docs_v2_77_100.log`).

| Result | First run | After the fixes |
|---|---|---|
| ✅ correct | 50 | **58** |
| 🟡 partial | 19 | 18 |
| ⚪ refusal | 20 | **11** |
| ❌ wrong | 11 | 13 |
| **Score** (✅ 2, 🟡 1, ⚪ 0, ❌ −1) | 108 | **121** |
| Few-shot leak / answer + refusal | 4 / 6 | **0 / 0** |
| Mean time per question | 21.3 s | ~22 s |

- Both fixes work: no `(ServerConfig)` anywhere, and answers like 24 (`e5be006`), 44, 80, 86 lost the contradictory refusal and became ✅.
- Side effect of the new example answer (one full sentence, no parenthetical): the model refuses less (20 → 11). Most of the new answers are right (9 reproducibility, 39 `--default-mm-loras`, 58 the full `lm_eval` command, 69 sharding at init, 88 RLHF, 95 `MAX_JOBS=6`), but two refusals became confident wrong answers: **73** (invents `vLLM_USE_V1=1` as a version; the right chunk was not in the context) and **96** (`response_parsed` instead of `message.parsed`). Same trade-off as always: fewer refusals, a few more inventions; the balance is clearly positive (+13).

**Network hang** — question 77 stayed asleep for minutes on an open HTTPS connection to Hugging Face (`ss` showed `ESTAB … :443`, 3 % CPU): `from_pretrained` asks the Hub for updates on every run, even with the model cached, and that request never answered. Fix: `Generator._from_pretrained` loads the model and the tokenizer with `local_files_only=True` first (no network request at all) and only downloads when they are not cached; a failed download is a clean `RagError`. Checked with no network at all (`unshare -rn`): `answer` works. The 24 questions after the fix ran without any hang.

## Step 20 — Mandatory part closed (2026-10-05)

**What**
- `answer` prints a `StudentSearchResultsAndAnswer` JSON; no model when no source matches. `answer_dataset` implemented: re-reads each source, loads the model once, tqdm.
- **Source paths of `answer_dataset` checked** (`CorpusReader.project_path`): a results JSON with `/etc/passwd` or `../../.ssh/id_rsa` is rejected **before PyTorch is loaded** (checked: 0.1 s, exit 1). Before, `PROJECT_ROOT / "/etc/passwd"` was simply `/etc/passwd`.
- **Symlinked corpus accepted:** `CorpusReader` normalises paths with `os.path.abspath` instead of `resolve()`, so `..` is still removed (`--raw_dir ../` rejected) but links are not followed. A `data/raw` linked outside the project (as on `goinfre`) is read and its paths keep the `data/raw/...` form (checked with a real link).
- `answer_dataset` with no questions writes an empty output without loading the model.
- Outdated tests fixed: `test_accepted` loaded Qwen for `answer` (it used the real index) and expected `NotImplementedError` from `answer_dataset`; `TestPendingCommands` removed (no pending command). Suite: 1,049 tests in ~20 s (was ~30 s, the model is never loaded now).
- README brought up to date (status, `answer` JSON example, `answer_dataset`, new Makefile rules, final prompt strategy, battery and docs-dataset results, robustness table).

## Step 21 — Bonuses 1 and 2: vector index and hybrid retrieval (2026-10-05 / 10-06) · commits `ceb60fe`, `ab988cb` (+ uncommitted)

**What**
- **Model chosen by measurement** (tables in README → *Semantic embeddings*): MiniLM needs ~400 s for the corpus here (~20 min on campus), over the 5-minute limit; `minishlab/potion-base-8M` (Model2Vec, static) embeds it in seconds with about the same recall as MiniLM at 128 tokens. With whole chunks, every model **lowered** recall when fused with BM25.
- **What made the vectors useful:** windows of ~400 characters with max-pooling per chunk (docs 0.44 → 0.70) and the file-path words in front of each window (code 0.343 → 0.404); together 0.73 / 0.596. Re-ranking the BM25 top N with `(1 − α) BM25 / best + α similarity` beats BM25 where list fusion (RRF) never did. Sweep N × α × query stopwords (48 cells): N = 30, α = 0.5 (centre of the plateau), query as written.
- **`VectorIndex`** (`src/vector_index.py`): `_get_window`, `_window_chunk` (path prefix), `build` (`np.repeat` for `owners`, `encode(..., normalize_embeddings=True)`), `save` / `load` as `vectors.npy` + `owners.npy`, `chunk_score` (`vectors @ query`, `np.maximum.at`), `chunk_count` property. The model is loaded from the local cache first and only on the first query (`_encoder`).
- **`load` without pydantic:** `np.load` already returns the arrays with their shape and type, so there is nothing like `model_validate_json`; the shapes are checked by hand (`ndim` before `len`). Missing file → `IndexNotFoundError`, garbage (`ValueError`) or arrays that do not fit → new `VectorIndexError` (`DatasetError` needs a pydantic `ValidationError`).
- **`HybridRetriever`** (`src/hybrid_retriever.py`): its own class, not inside `VectorIndex`, so the vector index does not depend on BM25. Same `search(query, k)` as `BM25Index`. Needs the id of each BM25 hit: `chunk_id` added to `SearchHit` (option A, cleaner than a `{id(chunk): position}` map). Checks `vector.chunk_count == bm25.chunk_count` (both exposed as properties, the attributes stay private). Returns `model_copy`s with the combined score; the stable sort keeps the BM25 order on ties.
- **CLI:** `index` builds and saves both indexes from the same chunks (no `--retriever` there: both are always built). `search`, `search_dataset` and `answer` get `--retriever` (`RetrieverMode = Literal["bm25", "hybrid"]`, default `hybrid`), chosen in one place (`RagCLI._load_retriever`). No service layer or `Retriever` base class yet: `BM25Index | HybridRetriever` is enough until bonuses 4 and 5.
- **Measured** (`make test_rag_base`, `make test_limits`): hybrid docs **0.910** / code **0.828** recall@5 (@10 0.920 / 0.859), exactly the experiment; `--retriever bm25` still 0.870 / 0.758. `index` 19.6–28.2 s, 199 questions 19.4 s. Checked against the subject: thresholds met with margin, and nothing forces a lexical default.
- **Tests:** `tests/conftest.py` replaces the embedding model for every test with a fake bag-of-words encoder (cached word slots, `np.add.at`), so `make test` never loads PyTorch for it. New `test_vector_index.py` and `test_hybrid_retriever.py` (44 tests). CLI tests that only save a BM25 index use `--retriever bm25`. Suite: 1,093 tests in ~60 s.

**Why**
- Lexical search misses paraphrases (the hard mini questions); the vectors catch some of them, but only as a re-ranker: the BM25 here is already tuned for this corpus.

**Learned**
- A static model averages its token vectors: on a 2000-character chunk the answer is diluted. Small windows + the best window per chunk fixed most of it.
- Removing stopwords helps BM25 but hurts the vectors: for an embedding the question words carry meaning.
- `np.maximum.at` (not `best[owners] = ...`): with repeated indexes a plain assignment keeps only the last value.
- The answer battery is not a retrieval measure: with the hybrid retriever question 07 came out reversed although its four right chunks were at the top. A 0.6B model reacts to the exact mix of chunks (as in the context-budget sweep). The battery took 521 s against 190 s; the hybrid part only adds ~6 s per `answer` (measured), so the rest was probably memory pressure (`auto` chose bfloat16).
- Real bugs caught while building it: `self.FILE_NAME` / `self.VECTORS_NAME` copied names in `save`, `owners.ndi`, a `)` that left `chunk_id` outside `SearchHit(...)`, `index` calling `_load_retriever` (load instead of build).

## Step 22 — `Retriever` base class and bonus 4, part 1: the embedding model without PyTorch (2026-10-06) · commit `0885414`

**What**
- **`make lint-strict`:** `np.ndarray` without type arguments replaced by `NDArray[np.float32]` / `NDArray[np.intp]` / `NDArray[np.float64]`. A test checked `index._model is fake_encoder`: mypy rejected it, and it did not prove what it said (a reload returned the same fake); it now counts the calls to `_load_model`.
- **`Retriever`** (`src/base_retriever.py`): abstract class with one method, `search(query, k)`. `BM25Index` and `HybridRetriever` inherit it; `VectorIndex` does not (it has no `search`: inheriting made it impossible to instantiate). `_load_retriever` and `_search_question` take a `Retriever` instead of `BM25Index | HybridRetriever`. Only `search` is abstract: `build` / `save` / `load` make no sense for `HybridRetriever`, which stores nothing of its own.
- **Bonus 4 plan** (caching: *"cache the index and query results to speed up cold start and repeated queries"*). Measured first: a hybrid search spent ~0.25 s starting Python, ~0.8 s loading `index.json` (0.45 read + 0.38 pydantic) and **~6.5 s importing `sentence_transformers` and loading the model**, then ~7 ms per question.
- **`StaticEncoder`** (`src/static_encoder.py`): `potion-base-8M` is a tokenizer plus a `(29528, 256)` table (`embeddings` in `model.safetensors`); a text is the normalised mean of its tokens' rows. Run with `tokenizers` + numpy: `from_pretrained` (cache first, download once, clean `RagError`), `encode`, `save` / `load` (`embeddings.npy`, `tokenizer.json`). On the 59,828 real windows: **same vectors as `vectors.npy`** (max difference 9e-8). Loads in ~0.3 s.
- `VectorIndex` uses it: no more `sentence_transformers` or `TYPE_CHECKING` import; `build` embeds by batches of 512 with a tqdm bar ("Embedding windows"); `save` writes the model next to the vectors and `load` reads it from there, so **searching never touches the Hugging Face cache**. `sentence-transformers` removed from the dependencies.
- **Measured** (`make test_limits`, then `evaluate` on its output): `index` **28.2 s → 13.2 s**, 199 questions **19.4 s → 5.4 s**, recall@1/3/5/10 **identical** (docs 0.910, code 0.828 @5). PyTorch is never imported by `index` or `search`.
- **Tests:** the `FakeEncoder` of `conftest.py` always normalises and has `save` (a marker file; the patched `StaticEncoder.load` returns the fake only for folders it saved into, so the `StaticEncoder` tests read real files). New `tests/test_static_encoder.py` (23 tests) with a hand-made 5-word `WordLevel` tokenizer and a table of known vectors. Suite: **1,116 tests**, `make lint-strict` clean.

**Why**
- The model was the cold start: 6.5 s of a ~7.3 s single search, for a model that is only a lookup table and a mean.

**Learned**
- `tokenizers` raises a plain `Exception` (a Rust error) for a missing or broken file, not `OSError` / `ValueError`: caught on its own, and the file existence checked first so a missing file still gives *"Run 'index' first"*.
- `np.save` adds `.npy` to any name: the tokenizer must be written with its own `save`, or it becomes `tokenizer.json.npy`.
- **Removing the averaging loop does not help** (measured three times each, corpus of 7.3 M tokens): loop 1.6 s, `np.add.reduceat` 13–16 s (it first copies every token row, ~7.5 GB moved in batches), a count matrix times the table 1.6 s. Each `table[ids].mean()` already adds in C; the work is the ~1.9 billion additions, not the 59,828 Python iterations. Tokenizing (2.4 s, Rust) is the larger part. Loop kept.
- Benchmarks on a shared machine jump around (load average 2–5): repeat them and compare within the same run.

## Step 23 — Bonus 4, part 2: query-result cache (2026-10-06) · commit `cefe27c` (in-memory part) + uncommitted

**What**
- **`CachedRetriever`** (`src/cached_retriever.py`), decorator pattern: it **is** a `Retriever` and **contains** another one (BM25 or hybrid), which do not change. It receives `load_inner`, a function that builds the wrapped retriever, instead of the retriever itself, and only calls it on the first question that is not cached (`_retriever()`, same idea as `VectorIndex._encoder`). Results are kept by `"<k>:<query>"`, and `search` returns copies.
- **On disk:** `QueryCache(namespace, entries)` in `models.py` (pydantic, not pickle). `_read()` starts empty when the file is missing, broken, not UTF-8 (`UnicodeDecodeError`, found in review: `'garbage {'` is ASCII and did not catch it) or from another namespace. `flush()` writes only if something was added (`_changed`, reset after a good write), to a `.tmp` file first and then `os.replace`; an `OSError` only prints a warning.
- **`Retriever.flush()`**, not abstract and empty: the commands call `indexer.flush()` without knowing which retriever they have (option (a), also ready for the HTTP API). Not called `save`: `BM25Index.save(index_dir)` would be an incompatible override.
- **Namespace** (`RagCLI._cache_namespace`): retriever, `K1`, `B` (+ `CANDIDATES`, `ALPHA`, model for hybrid) and `size:mtime_ns` of `index.json` (+ `vectors.npy`), read with `stat()` only. A missing file is `"<name>:missing"`: the "Run 'index' first" error still comes from loading. Kept in `RagCLI` so `CachedRetriever` stays generic.
- **CLI:** `_load_retriever(index_dir, retriever, cache)` → `_build_retriever` (the old body) or `CachedRetriever(lambda: _build_retriever(...), namespace, path)`. `--cache` (default `True`, off with `--cache=False`) in `search`, `search_dataset` and `answer`; `flush()` after the searches (after the loop in `search_dataset`, before loading Qwen in `answer`). `index` removes the caches after saving the new index.
- **Bug found while measuring:** one shared `query_cache.json` was overwritten at every switch of `--retriever` (a `bm25` search left 1 entry where 100 `hybrid` ones were). Fixed with one file per retriever: `FILE_NAME = "query_cache_{name}.json"`, `CachedRetriever.file_path(index_dir, name)`; `index` removes all of them (`get_args(RetrieverMode)`).
- **Also:** `HybridRetriever` asked BM25 for `CANDIDATES` (30) hits whatever `k` was, so `search --k 50` returned 30. Now `max(CANDIDATES, k)`, with a test that fails without it.
- **Tests:** `tests/test_cached_retriever.py` (33 tests, checked with 6 mutations of the code: each one breaks at least one test). Suite: **1,150 tests**, `make lint-strict` clean.

**Measured** (development PC; results and recall identical with and without cache):

| | No cache | Cached |
|---|---|---|
| `search_dataset` docs, hybrid | 1.94 s | 0.32 s |
| `search_dataset` docs, bm25 | 0.94 s | 0.33 s |
| `make test_retrieval` (199 questions) | 4.4 s | 0.7 s |
| `search`, one question | 0.78 s | 0.33 s |

Cache file: ~3.7 MB for 100 questions with k = 10 (whole `SearchHit`s, chunk text included).

**Why**
- Each CLI command is a new process, so an in-memory dict (or `lru_cache`) dies with it: the cache has to live on disk. Storing the whole `SearchHit` (and not only `chunk_id` + score) is what lets a cached command skip loading the index.

**Learned**
- Lazy loading is the whole point: loading the indexes (~1 s) costs far more than the searches (~7 ms each).
- A static method can read other classes' constants (`BM25Index.K1`) without any instance.
- `os.replace` is atomic: either the old file or the new one, never half of each.

## Step 24 — Bonus 5: choosing the HTTP framework (2026-10-06)

**Decision:** FastAPI + uvicorn, not the standard library's `http.server`.

**Why**
- **It fits the existing code.** The project already describes every input and output with pydantic models. FastAPI validates requests and serialises responses with those same models, so the API returns exactly the JSON the CLI writes (`StudentSearchResults`, `StudentSearchResultsAndAnswer`) without building it by hand.
- **Less code of my own, fewer places to get wrong.** Routing, request parsing, validation (automatic 422 errors) and status codes come with the framework. With `http.server` all of that would be hand-written.
- **Easy to demonstrate.** `/docs` gives an interactive page to try every endpoint from the browser during the defense.
- **Cost:** two dependencies (`fastapi`, `uvicorn`), allowed by the subject ("you may use any library you like"), plus `httpx` for the tests (`fastapi.testclient`).

**Plan:** a `RagApi` class (`src/api.py`) that keeps the retriever and the generator loaded between requests, and a `serve` command that starts it on `127.0.0.1` by default.

## Step 25 — Bonus 5: the HTTP API finished (2026-10-07) · commits `5a3ad92`, `9963fad`

**What**
- `RagApi` (`src/api.py`) with `/health`, `/search` and `/answer`, started by `serve` (`--host`, `--port`, `--index_dir`, `--retriever`, `--dtype`, `--cache`) or `make serve`. The endpoints return the same pydantic models as the CLI and save nothing.
- **`/answer` fixed.** My first version built a `StudentSearchResults` and then used it as the list of hits: `if results:` was always true (a pydantic model is always truthy), `for hit in results` walked the model's *fields*, and `_search_question` only keeps locations, not the chunk text the model needs. Now it calls `retriever.search` directly, like the `answer` command.
- **The model is loaded once**, on the first `/answer` (`_get_generator`), and kept in `self._generator`. Creating `Generator()` per request cost 20–40 s every time. `/health` and `/search` never load it.
- **`dtype` belongs to the server, not to each request.** I had put it as a parameter of `/answer`, but the model is loaded only once: a later request with another `dtype` would have been silently ignored. Back to `serve --dtype`.
- **Errors as HTTP codes, never a traceback.** An uncaught `RagError` inside an endpoint was a 500 with a traceback in the log: `/search` without an index did that, because `CachedRetriever` only loads the index on the first uncached search. Now one handler for the whole app (`self.app.exception_handler(RagError)(self._rag_error)`) turns every `RagError` into **503** `{"detail": "<message>"}`, the same message as the CLI. 503 = *Service Unavailable*: the request was fine, the server cannot serve it until something is fixed on its side. The server keeps running, and since the failed load is retried, running `index` in another terminal is enough (tested). PyTorch missing is also a 503 (`HTTPException`).
- **A query of only spaces is rejected** (422), as in the CLI: `StringConstraints(strip_whitespace=True, min_length=1)` instead of `Query(min_length=1)`, which counted the spaces. Spaces around a question are removed, as `str_strip_whitespace` does in the CLI.
- Invalid parameters give 422 (FastAPI, from the constraints). A second `serve` on a busy port exits with uvicorn's code 3 instead of the project's 1: left as is (no traceback, clear message).
- **Two locks (mutexes)**, see below.
- Docstrings (FastAPI shows them in `/docs`), a real `description`, the `serve` docstring.
- `tests/test_api.py`: 36 tests with `TestClient`, a small BM25 index and a fake generator. A failed load is simulated by putting a fake `src.generator` (or `None`, which makes the import fail) in `sys.modules`, so PyTorch is never imported. Suite: 1,187 tests.
- README: status, layout, command reference, Makefile, *HTTP API* section with `curl` examples, tests, design decisions, limitations.

**Why the locks**
- FastAPI runs every plain `def` endpoint in a thread pool: two requests at once run the same `RagApi` methods at the same time, on the same objects.
- `CachedRetriever` is not thread-safe: one thread adding to `_entries` while another serialises it in `flush` → `RuntimeError: dictionary changed size during iteration`; both writing the same `.tmp` file; the index loaded twice.
- The model: two first requests would load it twice (double memory, maybe `Killed`), and two generations at once double the memory too.
- `threading.Lock` is a mutex: one thread inside at a time, the others wait at the `with`. `with` releases it even on an exception (no deadlock from a forgotten `release`).
- **Two locks, not one:** `_retriever_lock` (search + flush, milliseconds) and `_generator_lock` (load + generate, seconds). With one, `/search` would wait for a whole generation. They are never held at the same time (the first is released before taking the second), so they cannot deadlock.

**Learned**
- A pydantic model is always truthy and iterating it gives `(field, value)` pairs: check the *list*, not the wrapper.
- State that is loaded once and shared between requests needs a lock as soon as the server has threads.
- An endpoint parameter that only matters the first time is a trap: it belongs to the server's configuration.
- `sys.modules[name] = None` makes `import name` raise `ImportError`: an easy way to test the "PyTorch missing" path without uninstalling anything.

## Step 26 — Bonus 3, part 0: where `index` spends its time (2026-10-07)

**What**
- Timed each phase of `index` with `time.perf_counter()` (one variable reset before each phase, printed to stderr; removed afterwards), on the real corpus (1,969 files, 20.1 M characters, 12,907 chunks):

| Phase | Time | Share of ~8.1 s |
|---|---|---|
| Read the files (`CorpusReader`) | 0.26 s | 3 % |
| Chunking (`PyChunker`, `MdChunker`) | 0.15 s | 2 % |
| Tokenize the chunks (`BM25Index.build`) | ~2.4 s | 30 % |
| Windows + embeddings (`VectorIndex.build`) | ~4.1 s | 50 % |
| Save both indexes | ~1.2 s | 15 % |

(The last three come from one 7.71 s measurement split with the tqdm bars.)

**Why it matters (for the defense)**
- Reading and chunking are almost free: comparing dates *before* reading is cleaner, but it is not where the time goes.
- The cost is **processing the chunks**: embeddings (~4 s) and BM25 tokenizing (~2.4 s). So "re-index only that file" must apply to **both** indexes. Rebuilding BM25 from all chunks would still re-tokenize the whole corpus, which is not what the subject asks.
- What stays proportional to the corpus, whatever is changed: loading the old index and saving the new one (~1.2 s).

**Learned**
- `time.perf_counter()`, not `time.time()`: a monotonic clock meant for durations (`time.time()` is wall-clock time and can jump).
- Measure before designing: my first proposal (rebuild BM25 whole, reuse only the vectors) would have left 30 % of the work untouched.

## Step 27 — Bonus 3: incremental indexing (2026-10-07)

**What**
- **Change detection without reading.** `CorpusReader.stamps()` gives `{path: FileStamp}` (size and `st_mtime_ns`) with one `stat` per file. `IncrementalIndexer._diff` compares it with the saved `manifest.json` and returns `CorpusChanges` (new, modified, deleted; a modified file is removed and added again). `CorpusReader.read(only=...)` reads only the new and modified files.
- **A fixed chunk order**, `Chunk.order_key()` = `(file_path, first_character_index)`, in the full index too. It used to be every `.py` and then every `.md`. Recall unchanged (only ties could move): docs 0.910 / code 0.828 hybrid, 0.870 / 0.758 bm25.
- **The plan.** `IncrementalIndexer._plan(old_chunks, to_remove, new_chunks)` returns an `IndexPlan`: the sorted chunks of the new index and `remap = {old id: new id}` of the kept ones; `fresh_ids` are the ids not reached by the remap, the only chunks to process. `order_key` works as the identity of a chunk, because a kept chunk and a new one never share it.
- **`BM25Index.update(plan)`**: lengths and postings moved with the remap without tokenizing, a term left without chunks removed, only the fresh chunks tokenized. IDF and the average length are recomputed by `__init__`. The tokenization of one chunk (`_chunk_tokens`) is shared with `build`.
- **`VectorIndex.update(plan)`**: the owners renumbered (`-1` for a removed chunk), a boolean mask keeps the rows, only the fresh windows are embedded (`_embed`, shared with `build`), then a stable `argsort` on the owners, which keeps the windows of a chunk in order.
- **`IncrementalIndexer.run(full)`** decides full or incremental, and `RagCLI.index` only prints its `IndexReport`. `index --full` forces a full index. The manifest is removed before anything is written and written last. The query caches are removed only when the index changes.
- Tests: `_diff`, `_plan`, both `update` methods compared with `build` (BM25 also on the real corpus), spies that check what is tokenized, embedded and read, and `run` on real files in `tmp_path`. Suite: 1,294 tests.

**Measured** (real corpus, whole command): full ~4.6 s, nothing changed ~0.8 s, 1 or 50 files ~1.35 s, 500 files ~2.0 s. After touching 140 files, the incremental index and a full one are identical (chunks, postings, lengths, owners, vectors).

**Bugs I made on the way (all silent: valid numbers, wrong results)**
- `fresh_ids` with `set(self.remap)`: the *keys* (old ids), not the values (new ids). Invisible when nothing shifts; caught with a new file in the middle.
- `enumerate(plan.remap)` instead of `plan.remap.items()`, and storing the id instead of `self._lengths[old_id]`: BM25 thought chunks were 0, 1, 2… tokens long. The test "`update` == `build`" caught it (`lengths` and `search` False).
- `self._owners[keep]` instead of `moved[keep]`: rows filtered but not renumbered, so two chunks shared an id.

**Learned**
- When several structures store positions, removing one element is a renumbering problem. Compute the map once and give it to everyone.
- The strongest test for an optimisation is equality with the slow path.
- Iterating a dict gives its keys; `.items()` gives the pairs; `enumerate` adds a counter, not the value.
- numpy: `array >= 0` is a mask with no loop, `array[mask]` keeps rows, `np.argsort(kind="stable")` gives the order without breaking ties.
- Write the "this is valid" marker (the manifest) last and remove it first: an interrupted run then falls back to the safe path.

## 📍 WHERE I STOPPED (2026-10-03)

The **retrieval** part is complete and above the targets (docs 0.87, code 0.758 recall@5). Next: **generation** (`Generator`, `answer`, `answer_dataset`). First step: `torch` CPU-only + `transformers` in `pyproject.toml`.

### ✅ Solved: decorator separated from its `def` (`PyChunker`)

Seen with a small chunk (size 90):
```python
class A:
    """Doc."""

    @property        ← ends the previous chunk
    def x(self):     ← starts the next one
```
**Cause:** `_find_cut` only accepts a separator in the second half of the window (minimum fill). When the decorator falls in the first half and the `def` in the second half, the `\n    @` cut is rejected and the next accepted separator is `\n    def `, so the cut lands between the decorator and its function. It also happens with stacked decorators (`@a\n@b\ndef`), comments right above a `def`, and decorated classes.

**Why it is not trivial:**
- `rfind` returns the **last** occurrence of each separator, and the separators are tried one by one: the cut does not "know" that the line above the chosen `def` is a decorator.
- Moving the cut back to before the decorator may break the minimum-fill rule (tiny chunk) or even move it before `start` (no progress → infinite loop).
- Several decorators, decorators with arguments over several lines (`@app.route(\n    "/x",\n)`) and comments make "the lines above the def" hard to define with simple string matching.

**Options to evaluate:**
1. After choosing a `def` cut, walk back line by line while the previous line starts with `@` (same indentation) and move the cut there, **only if** it stays after `start + size // 2`; otherwise keep the `def` cut.
2. Treat "decorators + def" as one block: search the `def` and then extend the cut to the first decorator line above it.
3. Use `ast`: decorated nodes give `decorator_list` with exact line numbers (needs the syntax-error fallback for the `.buildkite` file).
4. Accept it: with 2000-character chunks it is rare and, as measured, chunk boundaries barely change recall. Document it in the README.

Whatever is chosen: a test with `@property` / stacked decorators / a decorated class, and the chunker invariants must keep passing (no gaps, never above the max, always progresses).

**Solution (2026-10-02): option 2.** `PyChunker._find_cut` calls the parent cut and, if it lands at the start of a line that begins (after the indentation) with `@`, `def `, `async def ` or `class `, walks back over the single-line `@` lines just above it. Rules:
- it stops at the first line that is not a decorator (code and comments stay in the previous chunk);
- if the decorators reach back to `start` (the whole chunk would be decorators), the cut is **not** moved, so the chunk never ends up empty and the chunking always advances;
- the minimum fill can drop below half when moving back; decorator blocks are a few lines, so the chunk stays large.

Not covered: decorators whose arguments span several lines (`@app.route(\n    "/x",\n)`), since their last line does not start with `@`.

Measured on the real corpus (2000): chunks ending with a decorator line **406 → 1** of ~10,500; 12,881 → 12,907 chunks; recall unchanged (docs @5 0.86, code @5 0.717). Tests: the old `xfail` became `test_decorator_in_first_half_kept`, plus stacked decorators, method decorator, decorated class, non-decorator line stops, decorators from the chunk start, chunk starting mid-line.

---

## Next steps



**Housekeeping**
- [x] Docstrings on every module, class and function (`src/` and `tests/`).
- [ ] Real `README.md` (English, subject sections: Description, Instructions, Resources + AI use, System architecture, Chunking strategy, Retrieval method, Performance analysis, Design decisions, Challenges faced, Example usage). Current one is a personal reminder + commands + performance data.
- [ ] Keep TODOs here (they can be ticked), not only in commit messages.

**Reader follow-ups**

**Important**
- [ ] Symlinked corpus: `resolve()` follows symlinks, so if `data/raw` is a symlink (e.g. to `goinfre` on campus) the resolved folder is outside `PROJECT_ROOT` and `read()` rejects it — stricter than the subject. Fix: normalise with `os.path.abspath` (removes `..`, does not follow symlinks) instead of `resolve()` for the corpus folder and the file paths. `../` must still be rejected. Add a test with a symlinked `data/raw`.
- [x] Windows line endings — investigated, **no change**. `read_text` turns `\r\n` into `\n` (4 corpus files have `\r\n`, none is a source in the public datasets). The moulinette does not read the files: it only compares our numbers with the dataset's numbers, so what matters is how the dataset authors read them, and Python's default (`open`/`read_text`) is the most likely. Worst shift is 76 characters in a 3,520-character test file, 3 in the only one under `vllm/`: far inside the IoU 0.05 tolerance.

**Chunking (mandatory: two distinct strategies)**
- [x] Index convention: `last_character_index` is **exclusive**.
- [x] `Chunk(MinimalSource)` model, `CorpusReader`, `BaseChunker`, `FixedSizeChunker` + tests.
- [x] `PyChunker` (cut before definitions); single `_find_cut` with `CUT_BEFORE`.
- [x] `MdChunker` for `.md` / `.txt` (cut after code blocks; headings measured worse).
- [x] Choose the chunker by extension in `index`.
- [x] Tests for `PyChunker` and `MdChunker`.
- [x] Decorator separated from its `def` (see "Solved" above).
- [ ] Maybe: `index --chunking fixed|structural` (default structural) to compare strategies.

**Index, search and evaluation**
- [x] Own BM25 index + save/load + tests.
- [x] `index`, `search` (JSON), `search_dataset` + moulinette baseline.
- [x] `evaluate` (same recall as the moulinette, `--show_failed`) + tests.
- [x] Mini corpus + datasets in `tests/data`, `make test_rag`.

**Improve recall** (goal: docs ≥ 80 %, code ≥ 50 % recall@5 — met; more margin wanted for code)
- [x] Tokenizer: dotted names kept whole — measured, **removed** (no gain, more terms).
- [x] Tokenizer: keep whole `snake_case` identifiers too (second pass).
- [x] File path in the indexed text (enabled, measured).
- [x] Section headings / function names in the indexed text (after the structural chunkers).
- [x] Save which tokenizer built the index (in `IndexData`) and check it on `load`: a mismatch gives wrong results with no error.
- [ ] Try excluding `tests/`, `benchmarks/` (dataset code sources are all under `vllm/`).
- [x] Try chunk sizes, overlap, `K1`/`B`; record every result in README → Performance.
- [x] Query stopwords (`tokenize_query`).

**Generation (mandatory)**
- [x] Install `torch` CPU-only with uv (configured in `pyproject.toml`, so `uv sync` reproduces it) + `transformers`.
- [ ] Model cache on `goinfre` / `sgoinfre` (check which one on campus) via `HF_HOME`.
- [x] `Generator` class in its own module, imported **only** inside `answer` / `answer_dataset` (the other commands must not load torch). Chat template without thinking, token budget, max new tokens, clean output.
- [ ] `answer`: JSON output (`MinimalAnswer` in `StudentSearchResultsAndAnswer`), no-hits case, `index_dir` in the docstring.
- [ ] `answer_dataset` (reads results with `_open_file_save`, re-reads the source texts, one `Generator`, tqdm). Maybe save every N questions, so a crash does not lose everything.
- [x] Off-topic rule in `REMINDER`: refuses valid questions (Marlin, LoRA). Decide: shorter wording, drop it, or skip the model when the best BM25 score is very low.

**Memory-safe model loading — implemented 2026-10-03** (`Generator._get_available_ram`, `_choose_dtype`). Bugs found by testing: `"MemAvaiable:"` typo (always `None` → always bfloat16) and `dtype` passed positionally (`TypeError` in `Qwen3ForCausalLM.__init__`, so `answer` was broken). Verified with the real CLI: 9.4 GB → float32 (19 s, 5.0 GB peak); no meminfo → bfloat16 (42 s, 2.6 GB); 2 GB → clean error, exit 1, model not loaded. Pending: unit tests and docstrings (the `--dtype` flag and the English of the messages are done).

**Memory-safe model loading** (with too little RAM the kernel kills the process: `Killed`, exit 137, no Python exception, nothing to catch)
- [x] Measured peaks (`/usr/bin/time -f %M`): `answer` in **float32 ~5.1 GB**, in **bfloat16 ~3.0 GB** (both include the index and the bf16 → fp32 conversion while loading); `search` alone ~0.45 GB.
- [x] Read the available memory **before** loading the model. Nothing to install or import from a library and no sudo: `/proc/meminfo` is a plain text file readable by every user (`-r--r--r-- root`), read with `pathlib`, already imported in the project:
  ```python
  from pathlib import Path

  def available_gb() -> float:
      for line in Path("/proc/meminfo").read_text().splitlines():
          if line.startswith("MemAvailable:"):        # "MemAvailable:   9810268 kB"
              return int(line.split()[1]) / 1024 ** 2  # kB → GB
      raise ...                                        # line missing
  ```
  (OOP: as a method of a class, e.g. a static method of `Generator` or a small `MemoryCheck` class.)
- [x] Choose the dtype automatically: float32 if `available ≥ 5.5 GB`, else bfloat16 if `available ≥ 3.4 GB`, else a clean `RagError` (message + exit 1, no attempt to load). Thresholds = measured peak + ~0.4 GB margin (other programs can take memory while loading), as class constants (`FLOAT32_MIN_GB`, `BFLOAT16_MIN_GB`) with that explanation. Measured with `MAX_CONTEXT_TOKENS = 2500`: re-measure if the budget grows a lot. `answer_dataset` does not load the index (~0.45 GB less), so the same thresholds are conservative there.
- [x] `/proc/meminfo` only exists on Linux: catch `FileNotFoundError` / `OSError` and fall back to the conservative choice (bfloat16).
- [ ] Flag on `answer` (and `answer_dataset`) to force the dtype, **for testing only**. If the forced dtype does not fit, do not refuse, but print a warning to stderr (`float32 needs ~5 GB, only 2.1 GB available`).
- [x] Tests: the choice logic with fake values (no model loaded), the missing-file fallback, the error when nothing fits.
- [x] README: Requirements (automatic choice, thresholds) and Design decisions.

**Bonus** (only once the mandatory part is perfect)
- [x] Semantic embeddings (`potion-base-8M` instead of MiniLM: time limit) and hybrid retrieval (re-ranking instead of reciprocal rank fusion): Step 21.
- [x] `Retriever` base class (Step 22).
- [x] Caching: cold start (Step 22, model with numpy) and query-result cache (Step 23, `CachedRetriever`).
- [x] Local HTTP API (Steps 24–25: FastAPI + uvicorn, `serve`, `/health`, `/search`, `/answer`). Pending: practise the demo (`/docs` + `curl` from another terminal).
- [ ] Incremental indexing (bonus 3).

**Final**
- [ ] Edge cases, timings, README, `.gitignore` uncommented before delivery (no `data/`, PDF, `guia/`).

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
