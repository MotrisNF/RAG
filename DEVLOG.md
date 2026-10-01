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

## 📍 WHERE I STOPPED (2026-10-02)

The **mandatory** retrieval part is not complete yet: the subject requires two distinct chunking strategies.
1. `MarkdownChunker` for `.md` and `.txt`: another subclass with heading separators (`\n# `, `\n## `, `\n### `...) in `CUT_BEFORE`.
2. Choose the chunker by extension in `index` (`.py` → `PyChunker`, the rest → Markdown).
3. Tests for `PyChunker` and `MarkdownChunker` (reuse the `FixedSizeChunker` invariants).
4. Measure (docs, code, mini hard) and add the rows to README → Performance.

### ⚠️ To study: decorator separated from its `def` (`PyChunker`)

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

---

## Next steps

**Housekeeping**
- [x] Docstrings on every module, class and function (`src/` and `tests/`).
- [ ] Real `README.md` (English, subject sections: Description, Instructions, Resources + AI use, System architecture, Chunking strategy, Retrieval method, Performance analysis, Design decisions, Challenges faced, Example usage). Current one is a personal reminder + commands + performance data.
- [ ] Keep TODOs here (they can be ticked), not only in commit messages.

**Reader follow-ups**
- [ ] Symlinked corpus: `resolve()` follows symlinks, so if `data/raw` is a symlink (e.g. to `goinfre` on campus) the resolved folder is outside `PROJECT_ROOT` and `read()` rejects it — stricter than the subject. Fix: normalise with `os.path.abspath` (removes `..`, does not follow symlinks) instead of `resolve()` for the corpus folder and the file paths. `../` must still be rejected. Add a test with a symlinked `data/raw`.
- [x] Windows line endings — investigated, **no change**. `read_text` turns `\r\n` into `\n` (4 corpus files have `\r\n`, none is a source in the public datasets). The moulinette does not read the files: it only compares our numbers with the dataset's numbers, so what matters is how the dataset authors read them, and Python's default (`open`/`read_text`) is the most likely. Worst shift is 76 characters in a 3,520-character test file, 3 in the only one under `vllm/`: far inside the IoU 0.05 tolerance.

**Chunking (mandatory: two distinct strategies)**
- [x] Index convention: `last_character_index` is **exclusive**.
- [x] `Chunk(MinimalSource)` model, `CorpusReader`, `BaseChunker`, `FixedSizeChunker` + tests.
- [x] `PyChunker` (cut before definitions); single `_find_cut` with `CUT_BEFORE`.
- [ ] `MarkdownChunker` (cut before headings) for `.md` / `.txt`.
- [ ] Choose the chunker by extension in `index`.
- [ ] Tests for `PyChunker` and `MarkdownChunker`.
- [ ] Decorator separated from its `def` (see "To study" above).
- [ ] Maybe: `index --chunking fixed|structural` (default structural) to compare strategies.

**Index, search and evaluation**
- [x] Own BM25 index + save/load + tests.
- [x] `index`, `search` (JSON), `search_dataset` + moulinette baseline.
- [x] `evaluate` (same recall as the moulinette, `--show_failed`) + tests.
- [x] Mini corpus + datasets in `tests/data`, `make test_rag`.

**Improve recall** (goal: docs ≥ 80 %, code ≥ 50 % recall@5 — met; more margin wanted for code)
- [x] Tokenizer: dotted names kept whole (`llm.generate`). **Measure it.**
- [ ] Tokenizer: keep whole `snake_case` identifiers too. Today `max_model_len` → `max`, `model`, `len` only (the whole name is kept only inside a dotted name). Idea: a third pass `[a-z0-9]+(?:_[a-z0-9]+)+`.
- [ ] File path in the indexed text: code ready but commented in `BM25Index.build`. Uncomment, rebuild, measure.
- [ ] Section headings / function names in the indexed text (after the structural chunkers).
- [ ] Save which tokenizer built the index (in `IndexData`) and check it on `load`: a mismatch gives wrong results with no error.
- [ ] Try excluding `tests/`, `benchmarks/` (dataset code sources are all under `vllm/`).
- [ ] Try chunk sizes, overlap, `K1`/`B`; record every result in README → Performance.

**Generation (mandatory)**
- [ ] Install `torch` CPU-only with uv (configured in `pyproject.toml`, so `uv sync` reproduces it) + `transformers`.
- [ ] Model cache on `goinfre` / `sgoinfre` (check which one on campus) via `HF_HOME`.
- [ ] `Generator` class in its own module, imported **only** inside `answer` / `answer_dataset` (the other commands must not load torch). Chat template without thinking, token budget, max new tokens, clean output.
- [ ] `answer` (reuses `_search_question`) and `answer_dataset` (reads results with `_open_file_save`).

**Bonus** (only once the mandatory part is perfect)
- [ ] Semantic embeddings (`all-MiniLM-L6-v2`), hybrid retrieval (reciprocal rank fusion), incremental indexing, caching, local HTTP API.

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
