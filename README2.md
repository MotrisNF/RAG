*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine

> A Retrieval-Augmented Generation system that answers questions about the vLLM codebase, with its own BM25 search engine and a small local language model (Qwen3-0.6B), running entirely on CPU.

| Retrieval (recall@5, moulinette) | Indexing | Retrieval speed | Generation |
|---|---|---|---|
| **docs 87.0 %** (target 80 %) · **code 75.8 %** (target 50 %) | ~2 s for 1,969 files (limit 5 min) | < 1 s per 100 questions (limit 90 s for 200) | 16.6 s per answer on CPU · 76 % correct or partial on the docs dataset |

## Table of contents

1. [Description](#description)
2. [Instructions](#instructions)
3. [Example usage](#example-usage)
4. [System architecture](#system-architecture)
5. [Chunking strategy](#chunking-strategy)
6. [Retrieval method](#retrieval-method)
7. [Answer generation](#answer-generation)
8. [Performance analysis](#performance-analysis)
9. [Design decisions](#design-decisions)
10. [Challenges faced](#challenges-faced)
11. [Known limitations](#known-limitations)
12. [Resources](#resources)

---

## Description

A language model only knows what it was trained on. Asked about a specific codebase, a small model either says nothing useful or invents a confident answer. **Retrieval-Augmented Generation (RAG)** avoids retraining: at question time, the system finds the relevant pieces of the codebase and gives them to the model as context, with the instruction to answer only from them.

This project applies RAG to the [vLLM](https://github.com/vllm-project/vllm) repository, version 0.10.1: 1,969 Python, Markdown and text files, about 20 million characters. The pipeline has four stages:

1. **Indexing.** Every file is split into chunks of at most 2,000 characters, with one strategy for Python code and another for Markdown and text, and stored in an inverted index.
2. **Retrieval.** A question is matched against the index with **BM25**, a classic lexical ranking function implemented from scratch. The result is a ranked list of source locations: a file path and a character range.
3. **Augmentation.** The best chunks are placed in the prompt, within a fixed token budget.
4. **Generation.** **Qwen/Qwen3-0.6B** writes the answer from that context, or says that the context does not contain it.

The system is judged on two things:

- **Retrieval quality**, measured with recall@k against reference datasets. The targets are recall@5 ≥ 80 % on documentation questions and ≥ 50 % on code questions.
- **Answer quality**: coherent, on point and grounded in the retrieved sources.

Everything is exposed through a command-line interface built with Python Fire: `index`, `search`, `search_dataset`, `answer`, `answer_dataset` and `evaluate`.

---

## Instructions

### Requirements

| Requirement | Details |
|---|---|
| Python | ≥ 3.10 |
| Package manager | [uv](https://docs.astral.sh/uv/) (the only tool needed: `uv sync` installs everything) |
| Hardware | CPU only. PyTorch is installed from the CPU wheel index (configured in `pyproject.toml`), so no CUDA libraries are downloaded. |
| Memory for retrieval | ~0.5 GB (`index`, `search`, `search_dataset`, `evaluate`) |
| Memory for generation | The model type is chosen from the available memory before loading: **float32** (fastest) with ≥ 3.5 GB available, **bfloat16** (~2× slower) with ≥ 2.0 GB. Below that, `answer` stops with a clear error before loading anything. |
| Disk | ~1.2 GB for the model, downloaded on the first `answer` |

### Installation

```bash
uv sync            # or: make install
```

`uv sync --locked` installs exactly the versions of `uv.lock`, and fails if the lock file is out of date.

The first `answer` downloads Qwen3-0.6B to the Hugging Face cache. Later runs load it from the cache without any network request. To keep the cache out of a small home directory (for example on 42 computers):

```bash
export HF_HOME=/sgoinfre/students/$USER/hf
```

### Project layout

```
.
├── src/                      # the Python package: uv run python -m src <command>
├── tests/                    # pytest suite, mini corpus and answer battery
├── data/
│   ├── raw/vllm-0.10.1/                          # corpus to index
│   ├── processed/                                # index built by `index`
│   ├── datasets/
│   │   ├── UnansweredQuestions/                  # questions (input of search_dataset)
│   │   └── AnsweredQuestions/                    # questions + reference sources and answers
│   └── output/
│       ├── search_results/<DatasetScope>/              # output of search_dataset
│       └── search_results_and_answer/<DatasetScope>/   # output of answer_dataset
├── Makefile
├── pyproject.toml
└── uv.lock
```

`<DatasetScope>` is `UnansweredQuestions` or `AnsweredQuestions`, the folder the questions come from. The public datasets of both folders share file names, so each scope gets its own output folder and one run never overwrites the other.

### Running

```bash
uv run python -m src <command> [options]     # or: uv run -m src ...
uv run -m src <command> --help               # options of a command
```

Every input and output path is a CLI option with a default. Nothing is hard-coded.

### Command reference

| Command | Options (default) | Output |
|---|---|---|
| `index` | `--max_chunk_size` (2000, at most 2000) · `--raw_dir` (`data/raw`) · `--index_dir` (`data/processed`) | `index_dir/index.json` |
| `search <query>` | `--k` (10) · `--index_dir` (`data/processed`) | `StudentSearchResults` JSON on stdout |
| `search_dataset` | `--dataset_path` (required) · `--k` (10) · `--save_directory` (`data/output/search_results/UnansweredQuestions`) · `--index_dir` | `StudentSearchResults` JSON, same file name as the dataset |
| `answer <query>` | `--k` (10) · `--index_dir` · `--dtype` (`auto` \| `float32` \| `bfloat16`) | `StudentSearchResultsAndAnswer` JSON on stdout |
| `answer_dataset` | `--student_search_results_path` (required) · `--save_directory` (`data/output/search_results_and_answer/UnansweredQuestions`) · `--dtype` | `StudentSearchResultsAndAnswer` JSON, same file name as the input |
| `evaluate` | `--student_search_results_path` (required) · `--dataset_path` (required, an `AnsweredQuestions` file) · `--k` (10) · `--show_failed` | recall@1/3/5/10 report |

Progress bars and status messages go to stderr, so the JSON printed by `search` and `answer` can be redirected or piped (`> out.json`, `| jq`).

Boolean flags such as `--show_failed` go after the paths, or as `--show_failed=True`. Otherwise Fire takes the next word as the flag's value.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync` |
| `make run ARGS="..."` | `uv run -m src $(ARGS)` |
| `make debug ARGS="..."` | The same, under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the subject's flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | The pytest suite |
| `make test_rag` | index → search_dataset → evaluate on the mini corpus in `tests/data` (options `TEST_DATASET`, `TEST_K`, `TEST_CHUNK`) |
| `make test_rag_base` | The same on the real data, like the moulinette: k = 10, chunks of 2000, docs and code datasets |
| `make test_limits` | Checks the subject's time limits: indexing ≤ 5 min (`test_index`), 200 questions ≤ 90 s (`test_retrieval`) |
| `make test_answer` | Runs `answer` on the 10-question battery in `tests/answer_battery`, printing the expected answer after each one |
| `make test_answer_dataset` | The same on a whole `AnsweredQuestions` dataset, with a log (options `ANSWER_DATASET`, `ANSWER_START`, `ANSWER_LIMIT`, `ANSWER_ARGS`) |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and the `test_rag` outputs |
| `make cleandata` | Removes the index and the generated JSON files under `data/`, keeping the folder layout |

---

## Example usage

### Full pipeline, as the evaluation runs it

**1. Index the corpus once.**

```console
$ uv run python -m src index --max_chunk_size 2000
Read 1969 of 1969 files (20,147,166 characters) from data/raw
  .md: 178 | .py: 1763 | .txt: 28
12907 chunks (max 2000 characters)
Ingestion complete! Index saved under data/processed/
```

**2. Search a whole dataset.**

```console
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
File saved in 'data/output/search_results/UnansweredQuestions/dataset_docs_public.json'
```

**3. Score it with the moulinette.**

```console
$ ./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.610 Recall@3: 0.830 Recall@5: 0.870 Recall@10: 0.920
```

**4. Generate the answers.**

```console
$ uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
Torch mode: float32
Generating answers: 100%|██████████| 100/100
File saved in 'data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json'
```

Each entry of the output keeps the `question_id`, the question and the retrieved sources of step 2, and adds the answer:

```json
{
  "question_id": "17526382-…",
  "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
  "retrieved_sources": [
    { "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md", "first_character_index": …, "last_character_index": … }
  ],
  "answer": "The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`."
}
```

### Single questions

```console
$ uv run -m src search "How do I load a LoRA adapter?" --k 5
```

This prints a `StudentSearchResults` JSON with the 5 best sources.

```console
$ uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?"
```

This prints a `StudentSearchResultsAndAnswer` JSON whose answer is *"The minimum version of bitsandbytes required for vLLM quantization is 0.46.1."*

A question about a flag that does not exist (`--turbo-mode`), or one unrelated to the codebase ("What is the capital of France?"), gets the fixed refusal instead: *"The context does not contain the answer."*

### Own evaluation

`evaluate` computes the same recall as the moulinette, for iterating without it. `--show_failed` lists the questions whose sources were not found:

```console
$ uv run -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
    --show_failed
```

---

## System architecture

```
                 index                                     search / search_dataset
data/raw ──► CorpusReader ──► PyChunker (.py) ──┐                 │
                              MdChunker (.md,   ├─► BM25Index ──► save ──► data/processed/index.json
                                        .txt) ──┘   (Tokenizer)                    │
                                                                                   ▼
question ──► Tokenizer.tokenize_query ──► BM25Index.search (load) ──► top-k sources ──► JSON
                                                                          │
                              answer / answer_dataset                     ▼
                                                 chunk texts (from the index, or re-read from the files)
                                                                          │
                                                                          ▼
                                                     Generator (Qwen3-0.6B, loaded once)
                                                     context within the token budget → prompt
                                                     → greedy generation → answer → JSON
```

| Component | File | Role |
|---|---|---|
| `RagCLI` | `src/commands.py` | The six Fire commands. Each one validates its arguments, calls the components and writes the output. |
| `CliCommand.validated` | `src/commands.py` | Decorator: pydantic `validate_call` on every command (Fire ignores type hints) and a clean signature for `--help`. |
| `ArgvGuard` | `src/argv_guard.py` | Checks the command line **before** Fire runs it: unknown commands and options, private names, extra arguments, a misplaced `--help`. Fire would otherwise run the command first and complain afterwards. |
| `CorpusReader` | `src/reader.py` | Reads every `.py`, `.md` and `.txt` file under the corpus folder. Paths are stored relative to the project root, because the grader compares them verbatim. |
| `PyChunker`, `MdChunker` | `src/chunker.py` | The two chunking strategies, built on a shared `FixedSizeChunker`. |
| `Tokenizer` | `src/tokenizer_bm.py` | Text → terms, the same for chunks and queries. Stopwords are removed from queries only. |
| `BM25Index` | `src/bm25_index.py` | Builds the inverted index, ranks chunks, saves and loads it as JSON. |
| `Generator` | `src/generator.py` | Loads Qwen3-0.6B once, fits the chunks into the token budget, builds the prompt and generates the answer. |
| `Evaluator` | `src/evaluator.py` | recall@k with the moulinette's rules: same file, character IoU ≥ 0.05, sources of at most 2,000 characters. |
| Models | `src/models.py` | The subject's pydantic models (`MinimalSource`, `StudentSearchResults`…) and internal ones (`Chunk`, `Document`, `SearchHit`, `IndexData`). |
| Errors | `src/exceptions.py` | One hierarchy (`RagError`). Every expected failure ends with a clear message and an exit code, never a traceback. |

`answer` and `answer_dataset` import `Generator` inside the command. `index`, `search`, `evaluate` and `--help` never load PyTorch.

`answer_dataset` reads the output of `search_dataset`, which only stores the location of each source. It rebuilds each chunk's text from its file and character range, reading every file once. These are exactly the chunks the index stored, checked on 1,000 sources.

---

## Chunking strategy

Every file is split into chunks of at most `--max_chunk_size` characters: 2,000 by default, the moulinette's limit, and larger values are rejected. Offsets are exact: `text[first_character_index:last_character_index]` (end exclusive) gives back the chunk. Chunks never overlap, and only whitespace is left out.

### Common mechanism: `FixedSizeChunker`

1. The window `[start, start + size)` is cut at the best separator from a priority list, found with `rfind`.
2. A separator is only accepted **in the second half of the window**. This minimum fill avoids tiny chunks.
3. If no separator is found, the cut is hard.

The two strategies only change the separator list and `CUT_BEFORE`, the separators that start the next chunk instead of ending the current one. The size, offset and progress invariants are therefore proven once, for both.

### Python code: `PyChunker`

- **Separators, best first:** a top-level decorator, `class`, `def`, `async def`; then the same indented by four spaces (methods); then a blank line, a newline, a space.
- **Cut before definitions.** The cut goes before the keyword, so each chunk starts with a whole class, function or method when it fits. 51 % of its chunks start at a definition, against 27 % with plain fixed-size cuts.
- **Decorators stay with their definition.** A cut that lands on a definition is moved back over the `@decorator` lines just above it. Chunks ending with a decorator line went from 406 to 1, and the cut never moves before the start of the chunk, so chunking always advances.

### Markdown and text: `MdChunker`

- **Separators, best first:** the end of a fenced code block (`\n```\n`), then a blank line, a newline, a space.
- **Code blocks stay whole.** In the vLLM docs almost every opening fence has a language (```` ```bash ````, ```` ```python ````): 820 such openings against 2 bare ones. So `\n```\n` matches the closing fences, and a command or snippet stays whole in the chunk that closes it.
- **Headings are not separators.** 79 of the 100 docs reference sources start at a heading, yet every heading-based variant measured about 4 points lower (see [Performance analysis](#markdown-chunking-variants)).
- `.txt` files use the same strategy: the subject's "Markdown / text" chunking.

Result on the corpus: **12,907 chunks**.

---

## Retrieval method

**BM25**, implemented from scratch in `src/bm25_index.py`.

### Tokenizer

The text is lowercased and read in two passes:

1. Every run of letters or digits: `max_model_len` → `max`, `model`, `len`.
2. snake_case names kept whole, as an extra term: `max_model_len`. A question quoting an identifier matches it exactly, while the parts still match a question that only names one of them.

The **file path** of each chunk is tokenized too and added to its terms. 35 of the 99 code questions contain words (`keye`, `sampler`, `pynccl`) that appear only in the path of the right file.

### Index

An inverted index `term → {chunk id: term frequency}`, plus the length of each chunk in tokens. It is built once by `index`, saved as JSON and validated by pydantic (`IndexData`) on load. It loads in ~0.35 s.

### Ranking

For a query, every distinct term adds to the score of every chunk that contains it:

```
score(chunk) = Σ  idf(t) · tf · (K1 + 1) / (tf + K1 · (1 − B + B · length / avg_length))
idf(t)       = ln((N − df + 0.5) / (df + 0.5) + 1)
```

| Parameter | Value | Effect |
|---|---|---|
| `idf` | | Rewards rare terms. Never negative. |
| `K1` | 1.9 | Saturation: each repetition of a term helps less than the previous one. |
| `B` | 0.5 | Length normalisation: lowers the score of long chunks. |

Only chunks that share at least one term are scored. The top k are taken with `heapq.nsmallest` on `(−score, chunk id)`. The id breaks ties, so the ranking never depends on set iteration order and is identical between runs.

### Query stopwords

Question words and common English words (`what`, `how`, `the`, `use`…) are removed from the **query only**. In a code corpus these words are rare, so BM25 would give them a high idf (`why` 5.54, `how` 3.78, against 2.45 for `lora`), and chunks that only share the wording of the question would rank high. The index keeps every term, so the list can change without rebuilding it.

---

## Answer generation

### Context

The retrieved chunks are added best first, each as a block with a `[CHUNK i]` header, its file path and its text, until the budget of **2,200 tokens** (`MAX_CONTEXT_TOKENS`, counted with Qwen's own tokenizer). The first chunk that does not fit stops the loop, so the best chunks are always kept.

### Prompt

The prompt is designed for a 0.6B model, which follows examples better than long lists of rules.

| Part | Content |
|---|---|
| System message | Answer only from the context, quote the exact names and values, and when the answer is not there say the fixed sentence *"The context does not contain the answer."* |
| Two few-shot turns | One invented question answered by quoting the context (`max_retries = 3` in a fake `ServerConfig`), and one unrelated question refused. The examples are invented on purpose, so the model learns the behaviour, not the content. |
| User message | The context first, then the **question last**, where a small model attends most, then a one-line reminder: *"Answer the question using the context above."* |

### Decoding and post-processing

- **Greedy decoding** (`do_sample=False`): the same input always gives the same answer, so prompt variants can be compared fairly.
- **No thinking mode** (`enable_thinking=False`), at most 150 new tokens (`MAX_NEW_TOKENS`).
- **Refusal cleanup:** when the model gives an answer and then adds the refusal sentence, which contradicts it, the sentence is removed.

### Model loading

- **Loaded once per command.** `answer_dataset` loads the model once, not once per question.
- **Cache first.** The model is loaded with `local_files_only=True` and is only downloaded when it is not cached. This avoids a request to the Hugging Face Hub on every run, which once hung a run for minutes. A failed download is a clean error.
- **Model type from the available memory.** When Linux runs out of memory it kills the process (`Killed`, exit code 137) without any Python exception to catch. So the available memory is read from `/proc/meminfo` **before** loading:

| Available memory | Type | Why |
|---|---|---|
| ≥ 3.5 GB | float32 | ~3× faster: this CPU has no native bfloat16, so bfloat16 is emulated |
| ≥ 2.0 GB | bfloat16 | half the memory |
| < 2.0 GB | none: clean error, exit 1 | the model would not fit |
| unknown (no `/proc/meminfo`) | bfloat16 | the conservative choice |

`--dtype float32` or `--dtype bfloat16` forces a type for testing. If it looks too large for the available memory, a warning is printed, but the decision stays with the user.

---

## Performance analysis

All recall figures were measured on the public datasets (100 docs questions, 99 code questions) and confirmed with the moulinette. `evaluate` gives the same numbers. With about 100 questions, **one question ≈ 1 %**, so differences of one or two points are noise.

### Current results

Moulinette, chunks of 2,000 characters, k = 10:

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target (@5) |
|---|---|---|---|---|---|
| docs | 0.610 | 0.830 | **0.870** | 0.920 | ≥ 0.80 ✅ |
| code | 0.515 | 0.717 | **0.758** | 0.818 | ≥ 0.50 ✅ |

| Step | Time | Subject limit |
|---|---|---|
| `index`: read 1,969 files, chunk, build and save | ~2 s | 5 min |
| `search_dataset`: 100 questions, loading the index included | < 1 s | 90 s for 200 questions |
| Load the index | ~0.35 s | |
| Load Qwen3-0.6B | ~5–15 s | |
| One answer with `answer` (float32, ~2,000-token prompt, model load included) | ~20 s | none |
| `answer_dataset`, 100 questions (model loaded once) | 16.6 s per question, ~28 min in total | none |

`make test_limits` checks both time limits automatically.

### Evolution of recall@5

| Date | Change | docs | code |
|---|---|---|---|
| 2026-10-01 | Baseline: fixed-size chunks, plain tokenizer, `K1 = 1.5`, `B = 0.75` | 0.850 | 0.596 |
| 2026-10-02 | `PyChunker`, file path and snake_case terms, `K1 = 1.9`, `B = 0.5` | 0.830 | 0.727 |
| 2026-10-02 | `MdChunker` with the code-fence separator | 0.860 | 0.727 |
| 2026-10-02 | Dotted-name pass removed, `.txt` with `MdChunker`, decorators kept with their `def` | 0.860 | 0.717 |
| 2026-10-03 | Query stopwords | **0.870** | **0.758** |

Every change was measured on its own against the previous configuration, and kept only when it helped. The main experiments follow; the full record is in `DEVLOG.md`.

### Chunk size

| Chunk size | docs @5 | code @5 |
|---|---|---|
| **2000** | **0.850** | 0.596 |
| 1000 | 0.830 | **0.606** |
| 500 | 0.800 | 0.505 |

*Baseline configuration.*

- Small chunks are more precise, but they split answers.
- Big chunks contain the answer more often, but can fall below IoU 0.05 when the answer is short: a 2,000-character chunk containing a 40-character answer has IoU 0.02.

2,000 was kept: best on docs, and the most context per chunk for the generator.

### Ablation of the indexed terms

*Chunks of 2,000, k = 10, recall@5. Each row changes one thing.*

| Configuration | docs @5 | code @5 | Effect of that change |
|---|---|---|---|
| With path, snake_case and dotted passes | 0.830 | 0.727 | |
| − file path | 0.870 | 0.636 | path: **code +9**, docs −4 |
| − snake_case pass | 0.820 | 0.657 | snake_case: **code +7** |
| − dotted pass | 0.830 | 0.717 | neutral → removed |
| `K1/B = 1.5/0.75` | 0.820 | 0.727 | neutral |

The path hurts docs because code chunks take top-5 slots from the documentation: many code paths contain words of docs questions (`vllm/lora/…`, `…/quantization/…`). The code-fence Markdown chunker recovered that loss.

### Markdown chunking variants

| `.md` chunker | docs @5 |
|---|---|
| Plain fixed-size cuts | 0.860 |
| Heading separators (`#` > `##` > `###`) | 0.820 |
| One chunk per section | 0.820 |
| Headings + code-fence separator | 0.830 |
| **Code-fence separator only (implemented)** | **0.880** |

*Measured with the file name only in the indexed terms.*

The questions lost with headings were mostly "right file, IoU < 0.05": short sources whose chunk moved, sometimes just below the threshold (IoU 0.046, 0.048).

### Query stopwords

| Query filter | docs @5 | code @5 |
|---|---|---|
| None | 0.86 | 0.717 |
| Question words only (17) | 0.86 | 0.747 |
| **Question + common words (~45, implemented)** | **0.87** | **0.758** |
| NLTK English list (~150) | 0.86 | 0.758 |

A longer list does not help. A different list per dataset was not tried on purpose: the search does not know the type of the question.

### Discarded: chunk overlap

| Overlap | docs @5 | code @5 |
|---|---|---|
| 0 (implemented) | **0.86** | 0.717 |
| 200, all files | 0.83 | 0.737 |
| 400, all files | 0.80 | 0.747 |
| 200, `.py` only | 0.85 | 0.758 |
| 600, `.py` only | 0.85 | 0.727 |

On all files, two almost identical neighbouring chunks take two of the five slots. On `.py` only, the gain does not grow with the overlap, so it is close to noise.

### Generation

Answers cannot be scored automatically, so they were judged by hand.

**Battery of 10 questions** (`make test_answer`): 4 docs questions, 4 code questions, 1 off-topic question and 1 trap about a flag that does not exist. Each prompt change was run on the whole battery. Score: ✅ 2, 🟡 1, ⚪ refusal 0, ❌ −1.

| Prompt version | Score | Confident wrong answers |
|---|---|---|
| First prompt (4 rules) | — | evasive answers with the right context |
| Short prompt: "quote the exact names and values" | 7 | 4 |
| + off-topic rule | — | refused valid questions → dropped |
| + few-shot examples | 11 | 1 |
| + new reminder, budget 2,500 | 16 | 0 |
| **+ budget 2,200 (final)** | **17** | **0** |

**Context budget.** 1,500, 2,000, 2,500, 2,800 and 3,500 tokens were all tried. The answers only change when the *number of chunks* changes. More chunks means more distractors: with 3,500 the model copied values from the wrong chunk. 2,200 was the best.

**Sampling and thinking mode were both discarded.**
- Sampling refused more valid questions (4 → 6).
- Thinking mode was 3–7× slower (65–150 s per answer) and turned an honest refusal into a confident wrong answer.

**Full docs dataset with `answer_dataset`.** All 100 questions of `dataset_docs_public.json`, with the final code. Each answer was compared with the reference answer of the `AnsweredQuestions` dataset, and crossed with retrieval: is a correct source among the 10 retrieved ones?

| Result | Count | Correct source retrieved (92) | Correct source not retrieved (8) |
|---|---|---|---|
| ✅ correct | **54** | 54 | 0 |
| 🟡 partial or vague | 22 | 21 | 1 |
| ⚪ refusal | 11 | 7 | 4 (honest: the data was missing) |
| ❌ wrong | 13 | 10 | 3 |

- **76 %** of the answers are correct or partial, and **13 %** are wrong. Score (✅ 2, 🟡 1, ⚪ 0, ❌ −1): **117 / 200**.
- An earlier run of the same prompt, one `answer` call per question, scored 121 with 58 / 18 / 11 / 13. The difference is within the uncertainty of judging answers by hand.
- **Time: 16.6 s per question on average, ~28 min for the dataset**, against ~22 s per question with separate `answer` calls. The model is loaded once instead of once per question.

**Where the failures come from.** A correct source was retrieved in 92 of the 100 questions, so most of the loss is in generation, not retrieval:

| Failure | Questions | Examples |
|---|---|---|
| Over-refusal | 7 refusals with a correct source retrieved | "Where can I find…" questions (6, 10, 38), broad questions (4), short facts not picked up (47, 57, 74) |
| Value copied from a nearby but wrong chunk | most of the 10 wrong answers with a correct source retrieved | `FP4PerGroupSpec` instead of `quantization='quark'`, `encode` instead of `embed`, the quantization list of another hardware page |
| Answer off the point | 2 | how to *create* a profile instead of how to *view* it; how to *enable* a reasoning parser instead of how to *add* one |

When the correct source was not retrieved, the model did not make up an answer in 5 of the 8 questions: 4 refusals and 1 partial answer.

These are the "known reasoning limits" of a 0.6B model that the subject mentions.

### Tests

`make test` runs more than 1,000 pytest tests:

- **Chunkers:** exact offsets, no gaps or overlaps, never above the maximum, always progressing. Checked on random texts and on the whole real corpus.
- **BM25:** scores compared with a naive reference implementation of the formula, tie-breaking, save/load round trip.
- **Evaluator:** the same numbers as the moulinette.
- **CLI:** every command, argument validation, error messages and exit codes.
- **Mini corpus** in `tests/data`, with easy and hard questions, also run end to end by `make test_rag`.

---

## Design decisions

- **BM25 written from scratch** instead of a library. The inverted index, idf, saturation and tie-breaking are explicit and tested against the formula.
- **JSON index validated with pydantic** instead of pickle. It can be inspected, it loads in ~0.35 s, and a corrupted or foreign file gives a clear error instead of arbitrary behaviour.
- **Two chunkers on one cutting algorithm.** `PyChunker` and `MdChunker` only declare their separators, so the invariants are written and tested once.
- **One change at a time.** Every idea (path terms, tokenizer passes, heading cuts, overlap, stopwords, context budget, prompt wording) was measured alone. Heading cuts and overlap sounded right and were discarded by the numbers.
- **Stopwords on the query only**, so the list can change without rebuilding the index.
- **Paths relative to the project root**, written exactly as in the corpus (`data/raw/vllm-0.10.1/...`), because the grader compares them verbatim.
- **Lazy import and single load of the model.** Only `answer` and `answer_dataset` pay for PyTorch, and only once per command.
- **float32 on CPU when it fits.** It reads the prompt ~3× faster than bfloat16 (10 s against 29 s for ~2,000 tokens), at the cost of more memory. The type is chosen before loading, so a smaller machine gets a slower answer instead of a killed process.
- **Prompt for a small model:** few positive instructions, examples instead of rules, the question after the context, a fixed refusal sentence that is easy to detect, and greedy decoding for reproducible answers.
- **No traceback, ever.**
  - Arguments are validated by pydantic before a command runs.
  - The command line is checked before Fire runs it.
  - Every expected failure is a `RagError` with a message and an exit code: 1 for errors, 2 for usage, 130 for Ctrl+C.
- **Quality gates:** object-oriented code, type hints everywhere, docstrings on every module, class and function, and `flake8` and `mypy` clean with the subject's flags.

---

## Challenges faced

| Challenge | Solution |
|---|---|
| **Fire is permissive.** It ignores type hints, runs a command before rejecting unknown options or a late `--help`, and exposes private methods by name. | `CliCommand.validated` validates every argument with pydantic, and `ArgvGuard` checks the command line against the public commands before Fire runs anything. |
| **Questions and code rarely share words.** | snake_case names kept whole, the file path added to the indexed terms, and question words removed from the query. Code recall@5 went from 0.596 to 0.758. |
| **A fix that helps one dataset hurts the other.** The file path gave +9 on code and −4 on docs. | Found with a one-change-at-a-time ablation, and compensated by the code-fence Markdown chunker. |
| **Heading-based Markdown chunking was worse**, even though 79 % of the docs sources start at a heading. | Diagnosed as short sources falling just below IoU 0.05. The code-fence separator gave the best docs recall instead. |
| **Decorators separated from their function.** The minimum-fill rule rejected the decorator, so the cut fell between `@decorator` and `def`. | The cut is moved back over the decorator lines, without ever going before the start of the chunk. |
| **Rankings changed between runs.** Equal scores came out in `set` iteration order. | Ranking on `(score, chunk id)`. |
| **Slow generation**: 30–40 s per answer. | Profiling showed that bfloat16 is emulated on this CPU. Switched to float32 when it fits. |
| **Running out of memory cannot be caught**: the kernel kills the process. | The available memory is checked before loading: float32, bfloat16 or a clean error. |
| **Evasive answers with the right context.** | An *oracle test* (only the correct chunk in the context) separated retrieval failures from generation failures. A shorter prompt asking for exact values fixed them. |
| **"Paris" answered from the model's own knowledge.** An explicit off-topic rule fixed it, but made the model refuse valid questions. | Replaced by a few-shot refused example: off-topic questions are refused and valid ones are answered. |
| **The reminder leaked into the answers** ("The exact values are: …"), and so did the example's `(ServerConfig)`. | A neutral reminder, an example answer without a copyable tail, and removal of a refusal sentence that follows a real answer. |
| **A run hung for minutes** on a network request to the Hugging Face Hub, with the model already cached. | The model is loaded from the cache first, with `local_files_only=True`. |

---

## Known limitations

- **Lexical retrieval.** BM25 cannot match paraphrases that share no words with the answer. That is the target of semantic embeddings, which this version does not use.
- **A 0.6B model.** It sometimes picks a value from a similar but wrong chunk, and it is sensitive to the exact mix of chunks in its context.
- **Grounded is not always correct.** In one case the model quoted vLLM's own assertion message, which contradicts the condition next to it (`vllm/v1/core/kv_cache_coordinator.py`).
- **Memory check in containers.** The check reads the memory of the whole machine (`/proc/meminfo`). Inside a container or a cgroup with its own limit, it can overestimate what is available.

---

## Resources

### References

- P. Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020. The original RAG paper.
- S. Robertson, H. Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), 2009. The BM25 formula and its parameters.
- C. Manning, P. Raghavan, H. Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Cambridge University Press, 2008. Inverted indexes, tf-idf, evaluation with recall.
- N. Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), 2023. Why the position of information in the prompt matters.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). The chat template, the `enable_thinking` switch and the recommended decoding settings.
- Hugging Face Transformers: [chat templates](https://huggingface.co/docs/transformers/chat_templating) and [text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [uv: using PyTorch with uv](https://docs.astral.sh/uv/guides/integration/pytorch/). CPU-only wheels.
- Documentation of [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire) and [tqdm](https://tqdm.github.io/).

### Use of AI

An AI assistant was used throughout the project, as a tutor, a reviewer and a pair programmer:

- **Learning.** Explanations of the subject, RAG, BM25, tokenization, PyTorch and transformers, with exercises solved step by step (personal study notes, not part of the repository).
- **Review.** Reviews of my code (bugs found in the reader, the index, the evaluator, the CLI and the answer commands), of my design choices and of the Fire edge cases. It also checked the project against the subject's requirements.
- **Code, on request:**
  - the query stopword list;
  - most of the test suite;
  - several Makefile rules..
- **Documentation.** Drafts of docstrings and of this README.

Every part was reviewed, run and understood before being kept, and the design decisions were mine.
