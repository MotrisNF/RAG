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
