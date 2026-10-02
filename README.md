## It's a place-holder.

### So much important!!

I know i'm stupid, then don't forget to do this:

- Download the moulinnete. It's only at home.
- Download the rad data. It's only at home.

Please, don't be like always :)./m

---

## Commands (copy & paste)

Run everything from the project root.

### Quick test on the mini corpus (`tests/data`)

```bash
make test_rag
make test_rag TEST_DATASET=dataset_mini_hard.json
make test_rag TEST_DATASET=dataset_mini_hard.json TEST_CHUNK=2000 TEST_K=10
```

### index

```bash
# Defaults: data/raw -> data/processed, chunks of 2000 characters
uv run -m src index

# All options
uv run -m src index --max_chunk_size 2000 --raw_dir data/raw --index_dir data/processed
```

### search

```bash
# Defaults: k = 10, index in data/processed
uv run -m src search "How do I load a LoRA adapter?"

# All options
uv run -m src search "How do I load a LoRA adapter?" --k 5 --index_dir data/processed
```

### search_dataset

```bash
# Only the required option (k = 10, saved to data/output/search_results/UnansweredQuestions)
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json

# All options
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --index_dir data/processed
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions --index_dir data/processed
```

### evaluate

```bash
# Only the required options (k = 10)
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

# All options: --show_failed lists every question not found within k (like the moulinette's list_valid_questions)
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --show_failed
uv run -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --show_failed

# Failed questions of the mini hard dataset (run `make test_rag TEST_DATASET=dataset_mini_hard.json` first)
uv run -m src evaluate --student_search_results_path tests/data/output/search_results/dataset_mini_hard.json --dataset_path tests/data/datasets/AnsweredQuestions/dataset_mini_hard.json --k 5 --show_failed
```

### moulinette (official recall, for comparison)

```bash
# Recall@1/3/5/10
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

# PASS/FAIL against the subject thresholds (recall@5: docs 0.8, code 0.5)
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --threshold 0.8
./moulinette evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --threshold 0.5

# [VALID] / [INVALID] for every question (same list as `evaluate --show_failed`)
./moulinette list_valid_questions data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5
./moulinette list_valid_questions data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 5
```

---

## Performance

Measured on the public datasets (100 docs questions, 99 code questions) and on the mini datasets in `tests/data` (11 easy, 13 hard). Recall is computed by `evaluate`, which gives the same numbers as the moulinette (checked on every run below). With ~100 questions, **one question = ~1 %**: differences of a few points can be noise.

### Baseline (2026-10-01)

Configuration: `FixedSizeChunker` for every file (2000 characters, cut at `\n\n` > `\n` > space, only in the second half of the window), own BM25 (`K1 = 1.5`, `B = 0.75`), tokenizer `[a-z0-9]+` on lowercase text.

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Subject target (@5) |
|---|---|---|---|---|---|
| docs | 61.0 % | 78.0 % | **85.0 %** | 88.0 % | ≥ 80 % ✅ |
| code | 31.3 % | 53.5 % | **59.6 %** | 67.7 % | ≥ 50 % ✅ |

Questions not found within k = 5 (`evaluate --show_failed`, same list as the moulinette's `list_valid_questions`): 15 docs, 40 code.

### Timings

| Step | Time | Subject limit |
|---|---|---|
| `index` (read 1,969 files, 20.1 M characters, 12,134 chunks, build BM25, save) | ~1 s | 5 min |
| `search_dataset` (100 questions, load index + search) | ~1.5 s | 90 s for 200 questions |
| Load the saved index (32 MB JSON) | ~0.2 s | |
| One query | 4–14 ms | |

### Chunk size

| Chunk size | docs @5 | code @5 | mini easy @5 | mini hard @5 |
|---|---|---|---|---|
| 2000 | **85.0 %** | 59.6 % | 90.9 % | 76.9 % |
| 1000 | 83.0 % | **60.6 %** | | 92.3 % |
| 500 | 80.0 % | 50.5 % | 100 % | 76.9 % |

- Small chunks are more precise but split answers; big chunks contain the answer more often but may fall below IoU 0.05 when the answer is short (a 2000-character chunk containing a 40-character answer has IoU 0.02: not found).
- Ground-truth sources (median length): 878 characters for code (mean 821, max 1,578), 1,154 for docs (max 1,997).

### `PyChunker` vs `FixedSizeChunker` for `.py` files

`PyChunker` prefers to cut right before a definition (`@decorator`, `class`, `def`, `async def`, then indented methods). 51 % of its `.py` chunks start at a definition, against 27 % with `FixedSizeChunker`. Markdown/text files use `FixedSizeChunker` in both rows.

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
| 1000 | docs | Fixed | 0.620 | 0.790 | 0.830 | 0.840 |
| 1000 | docs | Py | 0.610 | 0.790 | 0.820 | 0.840 |
| 500 | docs | Fixed | 0.590 | 0.730 | 0.800 | 0.820 |
| 500 | docs | Py | 0.590 | 0.730 | 0.780 | 0.820 |
| 2000 | mini hard | Fixed / Py | 0.538 | 0.692 | 0.769 | 0.769 |
| 1000 | mini hard | Fixed / Py | 0.769 | 0.846 | 0.923 | 0.923 |
| 500 | mini hard | Fixed | 0.615 | 0.769 | 0.769 | 0.923 |
| 500 | mini hard | Py | 0.692 | 0.692 | 0.692 | 0.923 |

**Result: no clear gain.** At 2000 characters `PyChunker` improves recall@1 (+5 points) but lowers recall@5 (−5); at 500 it improves recall@5 (+2); at 1000 it is about the same.

**Why** (code, 2000 characters, k = 5: `PyChunker` gains 1 question and loses 6):
- 4 lost questions: the right file is still retrieved, but the chunk ranked first is another part of that file (IoU ≈ 0). Moving the cuts to the definitions puts the answer in a different chunk, which scores lower.
- 2 lost questions: the right file drops out of the top 5 (ranking changes).
- The ground-truth ranges are arbitrary windows (~880 characters), not aligned to functions, and IoU ≥ 0.05 is a very low bar: where exactly the cuts fall barely matters. What decides a hit is **which chunk ranks first**, i.e. the words it contains. Next levers for code: file path and identifiers in the indexed text (several lost questions name the module: *hunyuan*, *internvl*, *pynccl*).

`PyChunker` is kept: the subject requires a Python-specific strategy, and its effect on recall is neutral.

### Mini datasets (`make test_rag`, k = 5)

| Dataset | Chunk | @1 | @3 | @5 |
|---|---|---|---|---|
| easy | 500 | 1.000 | 1.000 | 1.000 |
| easy | 2000 | 0.909 | 0.909 | 0.909 |
| hard | 500 | 0.615 | 0.769 | 0.769 |
| hard | 2000 | 0.538 | 0.692 | 0.769 |

Hard questions missed (chunk 500): `hard-01`, `hard-02`, `hard-04`, all **paraphrases** (no or few words in common with the answer): the limit of a lexical method, target of the semantic-embeddings bonus. `hard-02` finds the right file first but not the right region (IoU).

### Ablation of the 2026-10-02 changes (chunk 2000, k = 10, recall@5)

Configuration measured ("current"): `PyChunker` for `.py`, `FixedSizeChunker` for `.md` / `.txt` (`MdChunker` written but not active), **full file path** added to the indexed terms, tokenizer with three passes (plain terms + dotted names + snake_case names), BM25 `K1 = 1.9`, `B = 0.5`. Each row removes or changes **one** thing from it.

| Configuration | docs @1 | docs @5 | code @1 | code @5 | mini hard @5 | Effect of that change |
|---|---|---|---|---|---|---|
| Baseline (2026-10-01) | 0.61 | 0.850 | 0.31 | 0.596 | 0.769 | |
| **Current** | 0.63 | **0.830** | 0.44 | **0.727** | 0.769 | |
| Current − file path | 0.58 | 0.870 | 0.39 | 0.636 | 0.769 | path: code +9, **docs −4** |
| Current − snake_case pass | 0.62 | 0.820 | 0.42 | 0.657 | 0.769 | snake_case: **code +7**, docs ±1 |
| Current − dotted pass | 0.65 | 0.830 | 0.48 | 0.717 | 0.769 | neutral |
| Current with `K1/B = 1.5/0.75` | 0.65 | 0.820 | 0.47 | 0.727 | 0.769 | neutral |
| Current with `FixedSizeChunker` for `.py` | 0.64 | 0.830 | 0.47 | 0.747 | 0.769 | noise (±2) |
| Current + `MdChunker` for `.md` | 0.61 | 0.770 | 0.45 | 0.727 | 0.769 | **docs −6** |

**Why the path lowers docs.** Adding the path only to `.py` chunks gives the same docs 0.830: the damage does not come from the paths of the Markdown files but from **code chunks taking top-5 slots from the documentation**. Many code paths contain words of docs questions (`vllm/lora/…`, `…/quantization/…`, `…/entrypoints/openai/…`), so every chunk of those files scores on them even when its text is unrelated.

**Which part of the path to index:**

| Indexed part of the path | docs @5 | code @5 |
|---|---|---|
| Full path (current) | 0.830 | 0.727 |
| **File name only** (`keye`, `fused_batched_moe`) | **0.860** | **0.717** |
| Parent folder + file name | 0.850 | 0.717 |
| No path | 0.870 | 0.636 |

The useful words of the path are usually in the **file name** (`keye`, `sampler`, `pynccl`); the ones that steal slots are usually **folders** (`lora`, `quantization`, `entrypoints`). File name only keeps almost all the code gain (−1 question) and recovers docs. 35 of the 99 code questions contain words that appear only in the path of their file, not in the text of the right chunk.

### Markdown chunking (with file name only in the indexed terms)

| `.md` chunker | docs @1 | docs @5 | code @5 |
|---|---|---|---|
| `FixedSizeChunker` | 0.61 | **0.860** | 0.717 |
| `MdChunker` (separators `\n###`, `\n##`, `\n#`, no space) | 0.60 | 0.820 | 0.717 |
| Headings with a space (`\n# ` > `\n## ` > `\n### `) | 0.63 | 0.820 | 0.717 |
| Headings with a space, deepest first (`\n### ` > `\n## ` > `\n# `) | 0.62 | 0.820 | 0.717 |
| One chunk per section (cut at every heading, long sections split like `FixedSizeChunker`) | 0.58 | 0.820 | 0.717 |
| One chunk per section, no path | 0.58 | 0.820 | 0.636 |

- Every heading-based variant loses ~4 points on docs against `FixedSizeChunker`.
- **79 of the 100 docs ground-truth sources start exactly at a heading**: the dataset was built by sections, so heading cuts should help in theory. Why they do not is still open (next idea: the interaction with `B`, since sections have very different lengths).
- Questions lost with heading cuts (one section per chunk vs fixed): mostly *right file, IoU < 0.05*: short sources (95–162 characters) whose chunk moved, sometimes just below the threshold (IoU 0.046, 0.048).
- Separators without a space (`\n#`) also match comment lines inside code blocks: the `.md` files have 1,287 real headings and 137 `# comment` lines in fenced code. No measurable effect on recall, but they should have the space.

**Decision so far:** index the file name only (best balance: docs 0.860 / code 0.717); keep the snake_case pass (main code gain); dotted pass and `K1/B` have no measurable effect. A Markdown-specific chunker is mandatory: with it docs is at 0.820, above the 0.80 target but with little margin, so it still needs work.

### Code-fence separator for Markdown (2026-10-02)

Idea: in technical docs the structure that breaks worst when chunking is the fenced code block (a command or a snippet split in two). New `MdChunker` separators: `\n```\n` (end of a code block, cut **after** it), then `\n\n`, `\n`, space; no heading separators, `CUT_BEFORE` empty.

In the vLLM docs almost every opening fence has a language (820 ` ```bash `/` ```python `… vs 2 bare ` ``` ` openings, 822 closings), so `\n```\n` matches closing fences: the block stays whole in the chunk it closes. Only 56 of 648 Markdown chunks (2000 characters) still end inside a code block (blocks longer than the chunk size).

With **file name only** in the indexed terms (experiment):

| `.md` chunker | docs @1 | **docs @5** | code @5 |
|---|---|---|---|
| `FixedSizeChunker` | 0.61 | 0.860 | 0.717 |
| Headings with a space | 0.63 | 0.820 | 0.717 |
| Headings + `\n```\n` | 0.62 | 0.830 | 0.717 |
| `\n```\n` first, then headings | 0.60 | 0.860 | 0.717 |
| **`\n```\n` only (no headings)** | 0.61 | **0.880** | 0.717 |

**Implemented** (`MdChunker` for `.md`, `PyChunker` for `.py`, `FixedSizeChunker` for `.txt`), measured with the moulinette on the real code as it is (full file path in the indexed terms, three-pass tokenizer, `K1 = 1.9`, `B = 0.5`):

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Before this change (@5) |
|---|---|---|---|---|---|
| docs | 0.640 | 0.820 | **0.860** | 0.910 | 0.830 |
| code | 0.465 | 0.677 | **0.727** | 0.798 | 0.727 |

12,881 chunks. Docs back above the baseline (0.850) with a 6-point margin over the 0.80 target; code keeps its gain (+13 points over the baseline). Heading cuts stay out: every heading variant measured lower.

### Tokenizer without the dotted pass, `.txt` with `MdChunker` (2026-10-02)

The dotted pass (`llm.generate`, `lora.md` kept whole) was removed: it only added terms and indexing time. `.txt` files now go to `MdChunker`, so there are exactly the two strategies of the subject (Python / Markdown-text); the 28 `.txt` files have no code blocks, so their chunks do not change. Moulinette, chunk 2000, k = 10:

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 |
|---|---|---|---|---|
| docs | 0.630 | 0.810 | **0.860** | 0.910 |
| code | 0.495 | 0.677 | **0.717** | 0.798 |

12,881 chunks. Code @5 drops 1 question (0.727 → 0.717), docs unchanged.

### Decorators kept with their definition (2026-10-02)

`PyChunker` moves a cut that lands before a `def`/`class`/decorator back over the `@` lines just above it. Chunks ending with a decorator line: **406 → 1** (of ~10,500 `.py` chunks); 12,907 chunks in total. Recall unchanged (docs @5 0.860, code @5 0.717): it makes chunks cleaner (and better context for the LLM), not easier to find.

### Chunk overlap (experiment, not implemented)

Each chunk starts `N` characters before the end of the previous one (aligned to a newline). Recall@5:

| Overlap | Chunks | docs @5 | code @5 | code @10 |
|---|---|---|---|---|
| 0 (current) | 12,881 | **0.86** | 0.717 | 0.798 |
| 200, all files | 14,340 | 0.83 | 0.737 | 0.798 |
| 400, all files | 16,067 | 0.80 | 0.747 | 0.838 |
| 200, `.py` only | 14,282 | 0.85 | **0.758** | 0.788 |
| 400, `.py` only | 15,931 | 0.86 | 0.747 | 0.838 |
| 600, `.py` only | 18,226 | 0.85 | 0.727 | 0.798 |

On all files it hurts docs: two almost identical neighbour chunks take two of the five slots. On `.py` only it gains 1–4 code questions, but the gain does not grow with the overlap (200 > 400 > 600), so it is close to noise (~1 question = 1 %). Not worth it for now: both targets are already met with margin, and it would break the "chunks never overlap" invariant.
