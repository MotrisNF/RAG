*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine

## Description

A **Retrieval-Augmented Generation (RAG)** system that answers questions about a real codebase: the [vLLM](https://github.com/vllm-project/vllm) repository, version 0.10.1 (1,969 Python, Markdown and text files, 20 million characters).

A small language model cannot know the details of a specific codebase, and when it does not know, it tends to invent a confident answer. A RAG system solves this in two steps:

1. **Retrieval**: find the few pieces of the codebase that are relevant to the question (here with **BM25**, a classic lexical ranking function implemented from scratch).
2. **Generation**: give those pieces to a small local model (**Qwen/Qwen3-0.6B**) as context, and ask it to answer **only** from them.

The project is judged on both parts: retrieval is measured with **recall@k** against reference datasets (targets: recall@5 ≥ 80 % on documentation questions and ≥ 50 % on code questions), and answers must be coherent, on point and grounded in the retrieved sources.

Current results (public datasets, recall@5): **docs 87 %, code 75.8 %**. Indexing takes ~2 s and each 100-question dataset is searched in under 1 s.

Everything is exposed through a command-line interface (Python Fire): `index`, `search`, `search_dataset`, `answer`, `answer_dataset` and `evaluate`.

> **Status (2026-10-03):** retrieval and the `answer` command are working. `answer_dataset` is in progress.

## Instructions

### Requirements

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).
- CPU only: PyTorch is installed from the CPU wheel index (configured in `pyproject.toml`), so no CUDA libraries are downloaded.
- ~5 GB of free RAM for answer generation (peak while loading the model in `float32`; `search` alone needs ~0.5 GB). With less memory the system swaps heavily or kills the process (`Killed`, exit code 137).

### Installation

```bash
uv sync            # or: make install
```

`uv sync --locked` installs exactly the versions of `uv.lock` and fails if the lock file is out of date.

The first `answer` downloads Qwen3-0.6B (~1.2 GB) to the Hugging Face cache. To keep it out of a small home directory (e.g. on 42 computers):

```bash
export HF_HOME=/goinfre/$USER/hf
```

### Data layout

```
data/
├── raw/vllm-0.10.1/                      # corpus to index
├── processed/                            # index built by `index`
├── datasets/
│   ├── UnansweredQuestions/              # questions (input of search_dataset)
│   └── AnsweredQuestions/                # questions + reference sources and answers
└── output/
    ├── search_results/<DatasetScope>/              # output of search_dataset
    └── search_results_and_answer/<DatasetScope>/   # output of answer_dataset
```

### Running

```bash
uv run python -m src <command> [options]     # or: uv run -m src ...
uv run -m src <command> --help               # options of a command
```

Every input and output path is a CLI option with a default; nothing is hard-coded.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync` |
| `make run ARGS="..."` | `uv run -m src $(ARGS)` |
| `make debug ARGS="..."` | Same, under `pdb` |
| `make lint` / `make lint-strict` | `flake8` + `mypy` (with the subject flags / `--strict`) |
| `make test` | The test suite (`pytest`) |
| `make test_rag` | Full pipeline (index → search_dataset → evaluate) on the mini corpus in `tests/data` |
| `make test_rag_base` | Same on the real data, like the moulinette (k = 10, chunks of 2000) |
| `make clean` / `make cleandata` | Remove caches / remove the generated index and outputs |

## Example usage

### Build the index

```bash
$ uv run -m src index --max_chunk_size 2000
Read 1969 of 1969 files (20,147,166 characters) from data/raw
  .md: 178 | .py: 1763 | .txt: 28
12907 chunks (max 2000 characters)
Ingestion complete! Index saved under data/processed/
```

### Search one question

```bash
$ uv run -m src search "How do I load a LoRA adapter?" --k 5
```

Prints a `StudentSearchResults` JSON with the 5 best sources (file path and character range). Progress messages go to stderr, so the JSON on stdout can be redirected or piped (`> out.json`, `| jq`).

### Search a whole dataset and evaluate it

```bash
uv run -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --show_failed
```

`evaluate` computes the same recall@1/3/5/10 as the moulinette; `--show_failed` lists the questions whose sources were not found (like the moulinette's `list_valid_questions`).

The same with the official grader:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
```

### Answer a question

```bash
$ uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?"
The minimum version of bitsandbytes required for vLLM quantization is 0.46.1.

$ uv run -m src answer "How do I enable the --turbo-mode flag in vLLM?"
The context does not contain the answer.
```

(`--turbo-mode` does not exist: the model refuses instead of inventing it.)

### Answer a whole dataset

```bash
uv run -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Reads the output of `search_dataset` and writes a `StudentSearchResultsAndAnswer` JSON (in progress, see Status).

## System architecture

```
                 index                                     search / search_dataset
data/raw ──► CorpusReader ──► PyChunker (.py) ──┐                 │
                              MdChunker (.md,   ├─► BM25Index ──► save ──► data/processed/index.json
                                        .txt) ──┘   (Tokenizer)                    │
                                                                                   ▼
question ──► Tokenizer.tokenize_query ──► BM25Index.search (load) ──► top-k chunks ──► JSON
                                                                          │
                                                       answer             ▼
                                                              Generator (Qwen3-0.6B)
                                                              _build_context → _build_messages
                                                              → chat template → generate → answer
```

| Component | File | Role |
|---|---|---|
| `RagCLI` | `src/commands.py` | The six Fire commands. Each one validates its arguments, calls the components and handles the output. |
| `CliCommand.validated` | `src/commands.py` | Decorator: pydantic `validate_call` on every command (Fire ignores type hints) and a clean signature for `--help`. |
| `ArgvGuard` | `src/argv_guard.py` | Checks `argv` **before** Fire runs: unknown commands and options, private names, misplaced `--help`. Fire would otherwise run the command first and complain afterwards. |
| `CorpusReader` | `src/reader.py` | Reads every `.py`, `.md` and `.txt` file under the corpus folder, with paths relative to the project root (the grader compares paths verbatim). Rejects folders outside the project. |
| `PyChunker`, `MdChunker` | `src/chunker.py` | The two chunking strategies, both built on `FixedSizeChunker` (see below). |
| `Tokenizer` | `src/tokenizer_bm.py` | Text → terms, the same for chunks and queries; drops stopwords from queries only. |
| `BM25Index` | `src/bm25_index.py` | Builds the inverted index, ranks chunks, saves and loads it as JSON. |
| `Generator` | `src/generator.py` | Loads Qwen3-0.6B once, fits the retrieved chunks into the token budget, builds the prompt and generates the answer. Imported only by `answer` / `answer_dataset`, so the other commands never load PyTorch. |
| `Evaluator` | `src/evaluator.py` | recall@k with the moulinette's rules (same file, character IoU ≥ 0.05, sources ≤ 2000 characters). |
| Models | `src/models.py` | The subject's pydantic models (`MinimalSource`, `StudentSearchResults`…) plus internal ones (`Chunk`, `Document`, `SearchHit`, `IndexData`). |
| Errors | `src/exceptions.py` | One hierarchy (`RagError`): every expected failure ends with a clear message and an exit code, never a traceback. |

## Chunking strategy

Every file is split into chunks of at most `--max_chunk_size` characters (default 2000, the moulinette's limit). Offsets are exact: `text[first_character_index:last_character_index]` (end exclusive) gives back the chunk, chunks never overlap, and only whitespace is left out.

**Common mechanism (`FixedSizeChunker`).** The window `[start, start + size)` is cut at the best separator from a priority list, found with `rfind`, but **only in the second half of the window**. This minimum fill avoids tiny chunks. If no separator is found, the cut is hard. Subclasses only change the separator list and `CUT_BEFORE`, the separators that must start the next chunk instead of ending the current one.

**Python code (`PyChunker`).** Separators, best first: top-level decorator, `class`, `def`, `async def`; then the same indented by four spaces (methods); then blank line, newline, space. Definitions are cut **before** their keyword, so each chunk starts with a whole class, function or method when it fits. A cut that lands on a definition is also moved back over the `@decorator` lines just above it, so a decorator is never separated from its function (chunks ending with a decorator line: 406 → 1).

**Markdown and text (`MdChunker`).** First separator: the end of a fenced code block (`\n```\n`), then blank line, newline, space. A command or snippet stays whole in the chunk that closes it. Headings are **not** separators: every heading-based variant measured 4 points lower (see Performance analysis). `.txt` files use the same strategy (the subject's "Markdown / text" chunking).

## Retrieval method

**BM25**, implemented from scratch (`src/bm25_index.py`).

**Tokenizer** (lowercase, two passes):
1. Every run of letters or digits: `max_model_len` → `max`, `model`, `len`.
2. snake_case names kept whole as an extra term: `max_model_len`. A question quoting an identifier matches it exactly, while the parts still match a question that only names one of them.

The **file path** of each chunk is tokenized too and added to its terms: 35 of the 99 code questions contain words (`keye`, `sampler`, `pynccl`) that are only in the file path, not in the text of the right chunk.

**Index.** An inverted index `term → {chunk id: term frequency}` plus the length of each chunk. It is built once by `index` and saved as JSON (`IndexData`, validated by pydantic on load).

**Ranking.** For a query, every distinct term adds to every chunk that contains it:

```
score(chunk) = Σ  idf(t) · tf · (K1 + 1) / (tf + K1 · (1 − B + B · length / avg_length))
idf(t)       = ln((N − df + 0.5) / (df + 0.5) + 1)
```

- `idf` rewards rare terms.
- The `tf` fraction saturates with `K1` (1.9): repeating a term helps less and less.
- `B` (0.5) lowers the score of long chunks.

Only chunks that share at least one term are scored. The top k are taken with `heapq.nsmallest` on `(−score, chunk id)`: the id breaks ties, so results never depend on set iteration order.

**Query stopwords.** Question words and common English words (`what`, `how`, `the`, `use`…) are removed from the **query only** (`Tokenizer.tokenize_query`). In a code corpus they are rare, so BM25 would give them a high idf (`why` 5.54, `how` 3.78, against 2.45 for `lora`), and chunks that only share the wording of the question would rank high.

**Generation.** The ranked chunks, best first, are added with their file path until `MAX_CONTEXT_TOKENS` (2500, counted with Qwen's own tokenizer). The prompt has a short system message (answer only from the context, quote exact names and values, a fixed sentence when the answer is not there) and a user message with the context first and the **question last**, followed by a one-line reminder. Generation is greedy (`do_sample=False`), without Qwen3's thinking mode, and limited to `MAX_NEW_TOKENS` (256).

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions) and on the mini datasets in `tests/data` (11 easy, 13 hard questions). `evaluate` gives the same numbers as the moulinette (checked on every run). With ~100 questions, **one question ≈ 1 %**, so differences of a few points can be noise.

### Current results

Moulinette, chunks of 2000 characters, k = 10:

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target (@5) |
|---|---|---|---|---|---|
| docs | 0.610 | 0.830 | **0.870** | 0.920 | ≥ 0.80 ✅ |
| code | 0.515 | 0.717 | **0.758** | 0.818 | ≥ 0.50 ✅ |

| Step | Time | Subject limit |
|---|---|---|
| `index` (1,969 files, 20.1 M characters → 12,907 chunks, build, save) | ~2 s | 5 min |
| `search_dataset` (100 questions, including loading the 35 MB index) | < 1 s | 90 s for 200 questions |
| Loading the index | ~0.35 s | |
| `answer`: load Qwen3-0.6B | ~5.6 s | |
| `answer`: one answer (prompt of ~2000 tokens) | ~12–17 s | |

### Evolution of recall@5

| Date | Change | docs | code |
|---|---|---|---|
| 2026-10-01 | Baseline: fixed-size chunks, plain tokenizer, BM25 `K1 = 1.5`, `B = 0.75` | 0.850 | 0.596 |
| 2026-10-02 | `PyChunker`, file path and snake_case terms, `K1 = 1.9`, `B = 0.5` | 0.830 | 0.727 |
| 2026-10-02 | `MdChunker` with the code-fence separator | 0.860 | 0.727 |
| 2026-10-02 | Dotted-name pass removed, `.txt` with `MdChunker`, decorators kept with their `def` | 0.860 | 0.717 |
| 2026-10-03 | Query stopwords | **0.870** | **0.758** |

All the experiments behind these steps follow, in the order they were made. Rows marked as not implemented were measured and discarded.

### Baseline (2026-10-01)

`FixedSizeChunker` for every file (2000 characters, cut at `\n\n` > `\n` > space, only in the second half of the window), BM25 `K1 = 1.5`, `B = 0.75`, tokenizer `[a-z0-9]+`.

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs | 0.610 | 0.780 | 0.850 | 0.880 |
| code | 0.313 | 0.535 | 0.596 | 0.677 |

Questions not found within k = 5: 15 docs, 40 code.

### Chunk size (baseline configuration)

| Chunk size | docs @5 | code @5 | mini easy @5 | mini hard @5 |
|---|---|---|---|---|
| **2000** | **0.850** | 0.596 | 0.909 | 0.769 |
| 1000 | 0.830 | **0.606** | | 0.923 |
| 500 | 0.800 | 0.505 | 1.000 | 0.769 |

- Small chunks are more precise but split answers.
- Big chunks contain the answer more often, but can fall below IoU 0.05 when the answer is short: a 2000-character chunk containing a 40-character answer has IoU 0.02, so it counts as not found.
- Reference sources have a median length of 878 characters for code and are up to 1,997 for docs.

2000 was kept: best on docs, and the largest context per chunk for the generator.

### `PyChunker` vs `FixedSizeChunker` for `.py` files

51 % of `PyChunker`'s chunks start at a definition, against 27 % with `FixedSizeChunker`. Markdown and text files use `FixedSizeChunker` in both rows.

| Size | Dataset | Chunker | @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|---|
| 2000 | code | Fixed | 0.313 | 0.535 | **0.596** | 0.677 |
| 2000 | code | Py | **0.364** | 0.515 | 0.545 | 0.636 |
| 1000 | code | Fixed | 0.394 | 0.535 | **0.606** | 0.687 |
| 1000 | code | Py | 0.374 | 0.525 | 0.596 | 0.646 |
| 500 | code | Fixed | 0.364 | 0.444 | 0.505 | 0.626 |
| 500 | code | Py | 0.364 | 0.455 | **0.525** | 0.616 |
| 2000 | docs | Fixed | 0.610 | 0.780 | **0.850** | 0.880 |
| 2000 | docs | Py | 0.580 | 0.790 | 0.840 | 0.870 |

**No clear gain on its own.** At 2000 characters it gains 1 code question and loses 6:
- In 4 of the lost questions, the right file is still retrieved, but another chunk of it ranks first.
- Reference ranges are arbitrary ~880-character windows, not aligned to functions, and IoU ≥ 0.05 is a low bar, so where exactly the cuts fall barely matters.
- What decides a hit is **which chunk ranks first**, that is, its words. That pointed to the next levers: the file path and identifiers.

`PyChunker` was kept because the subject requires a Python-specific strategy and it gives cleaner context to the generator.

### Mini datasets (`make test_rag`, k = 5, baseline configuration)

| Dataset | Chunk | @1 | @3 | @5 |
|---|---|---|---|---|
| easy | 500 | 1.000 | 1.000 | 1.000 |
| easy | 2000 | 0.909 | 0.909 | 0.909 |
| hard | 500 | 0.615 | 0.769 | 0.769 |
| hard | 2000 | 0.538 | 0.692 | 0.769 |

The hard questions missed are **paraphrases**, with few or no words in common with the answer. That is the limit of a lexical method, and the target of the semantic-embeddings bonus.

### Ablation of the 2026-10-02 changes (chunk 2000, k = 10, recall@5)

Configuration measured ("current" on that day): `PyChunker` for `.py`, `FixedSizeChunker` for `.md`/`.txt`, full file path in the indexed terms, three-pass tokenizer (plain + dotted names + snake_case), `K1 = 1.9`, `B = 0.5`. Each row changes **one** thing.

| Configuration | docs @1 | docs @5 | code @1 | code @5 | Effect of that change |
|---|---|---|---|---|---|
| Baseline (2026-10-01) | 0.61 | 0.850 | 0.31 | 0.596 | |
| **Current** | 0.63 | **0.830** | 0.44 | **0.727** | |
| − file path | 0.58 | 0.870 | 0.39 | 0.636 | path: **code +9**, docs −4 |
| − snake_case pass | 0.62 | 0.820 | 0.42 | 0.657 | snake_case: **code +7** |
| − dotted pass | 0.65 | 0.830 | 0.48 | 0.717 | neutral → later removed |
| `K1/B = 1.5/0.75` | 0.65 | 0.820 | 0.47 | 0.727 | neutral |
| `FixedSizeChunker` for `.py` | 0.64 | 0.830 | 0.47 | 0.747 | noise (±2) |
| + heading-based `MdChunker` for `.md` | 0.61 | 0.770 | 0.45 | 0.727 | **docs −6** |

**Why the path lowers docs.** The damage does not come from the paths of the Markdown files: adding the path only to `.py` chunks gives the same 0.830. It comes from **code chunks taking top-5 slots from the documentation**. Many code paths contain words of docs questions (`vllm/lora/…`, `…/quantization/…`), so every chunk of those files scores on them.

| Indexed part of the path | docs @5 | code @5 |
|---|---|---|
| Full path | 0.830 | 0.727 |
| File name only | 0.860 | 0.717 |
| Parent folder + file name | 0.850 | 0.717 |
| No path | 0.870 | 0.636 |

### Markdown chunking

| `.md` chunker (file name only in the indexed terms) | docs @1 | docs @5 | code @5 |
|---|---|---|---|
| `FixedSizeChunker` | 0.61 | 0.860 | 0.717 |
| Heading separators (`\n# ` > `\n## ` > `\n### `) | 0.63 | 0.820 | 0.717 |
| One chunk per section | 0.58 | 0.820 | 0.717 |
| Headings + code-fence separator | 0.62 | 0.830 | 0.717 |
| Code fence first, then headings | 0.60 | 0.860 | 0.717 |
| **Code-fence separator only (implemented)** | 0.61 | **0.880** | 0.717 |

- 79 of the 100 docs reference sources start exactly at a heading, yet every heading variant loses ~4 points.
- The questions lost are mostly *right file, IoU < 0.05*: short sources whose chunk moved, sometimes just below the threshold (IoU 0.046, 0.048).
- In the vLLM docs almost every opening fence has a language (```` ```bash ````, ```` ```python ````): 820 such openings against 2 bare ones, and 822 closings. So `\n```\n` matches the **closing** fences, and the block stays whole in the chunk it closes.
- Only 56 of 648 Markdown chunks still end inside a code block, the ones whose blocks are longer than the chunk size.

With the full path (the configuration kept) the code-fence `MdChunker` gives **docs 0.860 / code 0.727**.

### Tokenizer and `.txt` (2026-10-02)

- **Dotted pass removed.** It kept names like `llm.generate` whole. It had no measurable effect and only added terms: code @5 0.727 → 0.717, one question.
- **`.txt` files moved to `MdChunker`.** This leaves exactly the two strategies of the subject. The 28 `.txt` files have no code blocks, so their chunks did not change.

### Decorators kept with their definition (2026-10-02)

| | Before | After |
|---|---|---|
| `.py` chunks ending with a decorator line | 406 | 1 |
| Total chunks | 12,881 | 12,907 |
| Recall | | unchanged |

It makes chunks cleaner, which gives better context for the generator, not easier to find.

### Query stopwords (2026-10-03)

| Query filter | docs @1 | **docs @5** | docs @10 | code @1 | **code @5** | code @10 |
|---|---|---|---|---|---|---|
| None | 0.63 | 0.86 | 0.91 | 0.495 | 0.717 | 0.798 |
| Question words only (17) | 0.61 | 0.86 | 0.90 | 0.495 | 0.747 | 0.808 |
| **Question + common words (~45, implemented)** | 0.61 | **0.87** | 0.92 | 0.515 | **0.758** | 0.818 |
| NLTK English list (~150) | 0.60 | 0.86 | 0.93 | 0.525 | 0.758 | 0.818 |
| NLTK + extra (`use`, `get`, `need`…) | 0.60 | 0.85 | 0.92 | 0.515 | 0.758 | 0.818 |

- **Code gains 4 questions at @5.** A longer list does not help.
- **Docs @1 −2 is churn, not a pattern:** 9 questions lose their first hit and 7 gain one. In docs prose, question words are normal words ("How to…").
- **Not done on purpose:**
  - a different list per dataset, since the search does not know the question type;
  - adding words one by one while watching which questions improve, which would overfit the public datasets.

### Chunk overlap (not implemented)

| Overlap | Chunks | docs @5 | code @5 | code @10 |
|---|---|---|---|---|
| 0 (implemented) | 12,881 | **0.86** | 0.717 | 0.798 |
| 200, all files | 14,340 | 0.83 | 0.737 | 0.798 |
| 400, all files | 16,067 | 0.80 | 0.747 | 0.838 |
| 200, `.py` only | 14,282 | 0.85 | **0.758** | 0.788 |
| 400, `.py` only | 15,931 | 0.86 | 0.747 | 0.838 |
| 600, `.py` only | 18,226 | 0.85 | 0.727 | 0.798 |

- **On all files it hurts docs:** two almost identical neighbour chunks take two of the five slots.
- **On `.py` only the gain does not grow with the overlap** (200 > 400 > 600), so it is close to noise.

### Generation (2026-10-03)

**Speed.** Qwen3-0.6B is stored in `bfloat16`, and transformers loads it in that type by default. This CPU has no native bfloat16, so it is emulated:

| Model type | Read a ~2000-token prompt | Per generated token |
|---|---|---|
| bfloat16 (default) | 29 s | ~150 ms |
| **float32 (implemented)** | **10 s** | ~150 ms |

Reading the prompt grows with its length: 1,273 tokens take 6 s in float32. The subject sets no time limit for generation (the 90 s are for retrieval).

**Prompt.** The question *"What CUDA architectures are supported for Marlin MOE kernels in vLLM?"* (expected `8.0;8.7;9.0+PTX`, in `CMakeLists.txt`) was answered with *"…are specified in the context…"*, although retrieval was right: the correct chunk ranked 2nd.

| Variant | Result |
|---|---|
| 6 chunks, first prompt (4 rules: only the context, do not invent, refuse, cite the file) | ❌ describes the context |
| **Oracle**: only the correct chunk | ✅ |
| Reversed order (best chunk right before the question) | ❌ |
| 3 chunks, first prompt | ❌ invents `MARLIN_ARCHS` |
| 6 or 3 chunks, **short prompt: "Quote the exact names and values written in the context"** | ✅ `8.0;8.7;9.0+PTX` |

The oracle test separates retrieval failures from generation failures. With the right chunk alone the model answers; with five more it does not. Reordering or trimming the context did not help, but a shorter, positive instruction did.

| Question | First prompt | Short prompt |
|---|---|---|
| Minimum bitsandbytes version | ✅ 0.46.1 | ✅ 0.46.1 |
| `--turbo-mode` flag (does not exist) | ✅ refuses | ✅ refuses |
| How to load a LoRA adapter | ✅ (verbose) | ✅ (direct) |
| Quantization methods for Mixtral | ❌ vague | ❌ exact list, but from the wrong chunk |
| Capital of France | ❌ "Paris" | ❌ "Paris" |

**Off-topic rule.** A second sentence was added to the reminder: if the question is not about vLLM, LLMs, model serving or programming, answer with the fixed refusal. Judging the topic of a question should be easier for a small model than checking whether the context contains the answer. Measured with the current code (k = 10):

| Question | Short prompt | + off-topic rule |
|---|---|---|
| Capital of France | ❌ "Paris" | ✅ refuses |
| `--turbo-mode` flag (does not exist) | ✅ refuses | ✅ refuses |
| Minimum bitsandbytes version | ✅ 0.46.1 | ✅ 0.46.1 |
| Marlin MOE architectures | ✅ `8.0;8.7;9.0+PTX` | ❌ **refuses** |
| How to load a LoRA adapter | ✅ | ❌ **refuses** |

The rule fixes out-of-domain questions, but makes the model refuse valid ones: a longer reminder dilutes "quote the exact values", and questions that do not name vLLM ("How do I load a LoRA adapter?") look off-topic. A refused valid question is worse than a wrong answer to an out-of-domain one, so this trade-off is still open (next options: a shorter wording, or skipping the model when the best BM25 score is very low). The other remaining failures (choosing the wrong list among similar chunks) are the "known reasoning limits" of a 0.6B model mentioned in the subject.

### Tests

`make test` runs **1,030 tests** in ~30 s:

- **Chunkers:**
  - invariants (exact offsets, no gaps or overlaps, never above the maximum, always progresses) on random texts and on the whole real corpus;
  - separator priorities;
  - code blocks kept whole;
  - decorators kept with their definition.
- **Tokenizer and BM25:**
  - scores compared with a naive reference implementation of the formula;
  - idf, saturation, length normalisation, deterministic tie-breaking;
  - save/load round trip;
  - query stopwords.
- **Evaluator:** IoU, cutoffs, missing and unknown questions, validation errors, the same numbers as the moulinette.
- **CLI:** every command, argument validation and error messages, unknown or private options, exit codes, `--help`.
- **Mini datasets** in `tests/data`: a small vLLM-like corpus with easy and hard questions, also run end to end by `make test_rag`.

## Design decisions

- **BM25 written from scratch** instead of a library: the inverted index, idf, saturation and tie-breaking are explicit and testable against the formula, and the index format is our own pydantic model.
- **JSON index validated with pydantic** instead of pickle. It loads in ~0.35 s, can be inspected, and a corrupted or foreign file gives a clear error instead of arbitrary behaviour.
- **Two chunkers sharing one cutting algorithm.** `PyChunker` and `MdChunker` only declare their separators. The invariants are proven once, for every strategy.
- **Measure every change on its own.** Each idea (path terms, tokenizer passes, heading cuts, overlap, stopwords) was measured alone against the previous configuration, and kept only if it helped. Heading cuts and overlap sounded right and were discarded by the numbers.
- **Stopwords on the query only.** The index keeps every term, so the list can change without rebuilding it.
- **Lazy import of the generator.** `answer` and `answer_dataset` import `Generator` inside the command, so `index`, `search`, `evaluate` and `--help` never pay the seconds PyTorch takes to load.
- **Load the model once.** `Generator.__init__` loads Qwen; each question only calls `generate`.
- **float32 on CPU**: 3× faster to read the prompt than the default bfloat16, at the cost of more RAM (peak ~5.1 GB against ~3.0 GB).
- **Prompt for a small model:**
  - a short system message with positive instructions;
  - the question after the context, where a small model attends most;
  - the key rule repeated right before the answer;
  - a fixed refusal sentence, easy to detect;
  - greedy decoding, so answers are reproducible.
- **100 % object-oriented**, pydantic models for every exchanged structure, `flake8` + `mypy` clean, and a single error hierarchy so that every expected failure ends with a message and an exit code instead of a traceback.

## Challenges faced

- **Fire's permissiveness.** Fire ignores type hints, runs a command before rejecting unknown options or a late `--help`, turns a bare flag into `True`, and exposes private methods by name. Solved with `CliCommand.validated` (pydantic validation of every argument) and `ArgvGuard` (checks `argv` against the public commands before Fire runs anything).
- **Lexical retrieval on code.** Questions and code rarely share words. Three measured fixes:
  - snake_case names kept whole;
  - the file path added to the indexed terms;
  - question words removed from the query.

  Code recall@5 went from 0.596 to 0.758.
- **Fixes that help one dataset and hurt the other.** The full file path gave +9 code points and −4 on docs, because code chunks took documentation slots. Found with a one-change-at-a-time ablation, and compensated by the code-fence Markdown chunker.
- **Heading-based Markdown chunking was worse**, even though 79 % of the docs reference sources start at a heading. The lost questions were short sources falling just below IoU 0.05. The code-fence separator gave the best docs recall instead.
- **Decorators separated from their function.** The minimum-fill rule rejects a decorator in the first half of the window, so the cut fell between `@decorator` and `def`. Solved by moving the cut back over the decorator lines, without ever going before the start of the chunk. That guarantees the chunking always advances.
- **Non-deterministic rankings.** Equal scores came out in a different order between runs, because of `set` iteration order. Solved by ranking on `(score, chunk id)`.
- **Slow generation.** 30–40 s per answer, until profiling showed that bfloat16 is emulated on this CPU. Switched to float32.
- **Evasive answers with a correct context.** Diagnosed with the oracle test (only the correct chunk) and solved with a shorter prompt that asks for the exact values.
- **Refusing out-of-domain questions without refusing valid ones.** The 0.6B model answers "Paris" from its own knowledge despite the rules. An explicit off-topic rule fixes it, but also makes it refuse some valid questions: still being tuned.

## Resources

### References

- P. Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020: the original RAG paper.
- S. Robertson, H. Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), 2009: the BM25 formula and its parameters.
- C. Manning, P. Raghavan, H. Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Cambridge University Press, 2008: inverted indexes, tf-idf, evaluation with recall.
- N. Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), 2023: why the position of information in the prompt matters.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): chat template and the `enable_thinking` switch.
- [Hugging Face Transformers: chat templates](https://huggingface.co/docs/transformers/chat_templating) and [text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [uv: using PyTorch with uv](https://docs.astral.sh/uv/guides/integration/pytorch/): CPU-only wheels.
- [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [tqdm](https://tqdm.github.io/) documentation.

### Use of AI

An AI assistant was used throughout the project, as a tutor, a reviewer and a pair programmer:

- **Learning.** Explanations of the subject, RAG, BM25, tokenization, embeddings, PyTorch and transformers, with exercises solved step by step. These notes are kept in `guia/`.
- **Review.** Reviews of my code (bugs found in the reader, the index, the evaluator and the CLI), of my design choices and of the Fire edge cases.
- **Code, on request:**
  - the CLI hardening (`CliCommand.validated`);
  - The decorator handling of `PyChunker`;
  - the query stopword list;
  - most of the test suite.
- **Experiments.** Scripts for the ablations and measurements reported above (chunk sizes, path terms, Markdown variants, overlap, stopwords, model types, prompt variants), and the analysis of their results.
- **Documentation.** Drafts of the docstrings and of this README.

Every part was reviewed, run and understood before being kept, and the design decisions were mine.