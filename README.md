*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine

## Description

A **Retrieval-Augmented Generation (RAG)** system that answers questions about a real codebase: the [vLLM](https://github.com/vllm-project/vllm) repository, version 0.10.1 (1,969 Python, Markdown and text files, 20 million characters).

A small language model cannot know the details of a specific codebase, and when it does not know, it tends to invent a confident answer. A RAG system solves this in two steps:

1. **Retrieval**: find the few pieces of the codebase that are relevant to the question. Here with **BM25**, a classic lexical ranking function implemented from scratch, whose best results are then **re-ranked by meaning** with a small CPU embedding model (`minishlab/potion-base-8M`).
2. **Generation**: give those pieces to a small local model (**Qwen/Qwen3-0.6B**) as context, and ask it to answer **only** from them.

The project is judged on both parts: retrieval is measured with **recall@k** against reference datasets (targets: recall@5 ≥ 80 % on documentation questions and ≥ 50 % on code questions), and answers must be coherent, on point and grounded in the retrieved sources.

Current results:
- **Retrieval** (public datasets, recall@5): **docs 91 %, code 82.8 %** with the hybrid retriever (default), 87 % / 75.8 % with BM25 alone. Indexing takes ~13 s and each 100-question dataset is searched in ~3 s.
- **Generation** (the 100 public docs questions, judged by hand against the reference answers): **58 correct, 18 partly correct, 11 refusals, 13 wrong**, ~22 s per answer on CPU.

Everything is exposed through a command-line interface (Python Fire): `index`, `search`, `search_dataset`, `answer`, `answer_dataset` and `evaluate`.

> **Status (2026-10-06):** mandatory part complete: the six commands work end to end. Bonuses 1 (semantic embeddings) and 2 (hybrid retrieval) done; `--retriever bm25` keeps the lexical search alone. Bonus 4 (caching) in progress: the embedding model is saved with the index and run with numpy, so searching no longer loads PyTorch (cold start ~7 s → ~1 s); the query-result cache comes next.

## Instructions

### Requirements

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).
- CPU only: PyTorch is installed from the CPU wheel index (configured in `pyproject.toml`), so no CUDA libraries are downloaded.
- Free RAM for answer generation: **≥ 5.5 GB** to run the model in `float32` (fastest), **≥ 3.4 GB** for `bfloat16` (~2× slower). The type is chosen automatically from the available memory; with less than 3.4 GB, `answer` stops with a clear error before loading anything. `search` alone needs ~0.5 GB.

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
| `make test_index` / `make test_retrieval` / `make test_limits` | Check the subject time limits (indexing ≤ 5 min, 200 questions ≤ 90 s): prints the time and PASS / FAIL |
| `make test_answer` | `answer` on the 10 questions of `tests/answer_battery/`, each followed by its expected answer |
| `make test_answer_dataset` | `answer` on every question of an AnsweredQuestions dataset, with its reference answer; the output is saved to `data/output/answer_logs/` |
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
$ uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --k 5
```

Prints a `StudentSearchResultsAndAnswer` JSON with one entry: the sources given to the model and its answer (abridged):

```json
{
  "search_results": [
    {
      "question_id": "…",
      "question": "What is the minimum version of bitsandbytes required for vLLM quantization?",
      "retrieved_sources": [
        {"file_path": "data/raw/vllm-0.10.1/vllm/model_executor/layers/quantization/bitsandbytes.py", "first_character_index": 15340, "last_character_index": 16778},
        …
      ],
      "answer": "The minimum version of bitsandbytes required for vLLM quantization is `bitsandbytes >= 0.46.1`."
    }
  ],
  "k": 5
}
```

Progress messages (`Loading PyTorch...`, `Torch mode: float32`) go to stderr, so the JSON can be redirected. Other answers:

- *"How do I enable the --turbo-mode flag in vLLM?"* → `The context does not contain the answer.` (the flag does not exist: the model refuses instead of inventing it).
- A query that matches nothing (`"???"`, only stopwords) → `No relevant sources were found in the corpus.`, without loading the model.

### Answer a whole dataset

```bash
uv run -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Reads the output of `search_dataset`, re-reads the text of every retrieved source, loads the model **once** and answers each question with a progress bar, then writes a `StudentSearchResultsAndAnswer` JSON with the same file name. ~22 s per question on CPU (~35 min for 100). Every source path is checked before PyTorch is loaded: a path outside the project (`/etc/passwd`, `../..`) is rejected, and a file without questions is written without loading the model.

### Every option, with and without `make`

Each line below sets one option to show how it is written; the others keep their default. With `make`, the whole command goes inside `ARGS`. Use single quotes outside when the question has spaces, so the inner double quotes reach the program. Boolean flags such as `--show_failed` go **after** the paths (or use `--show_failed=True`). Otherwise Fire reads the next word as the flag's value.

#### `index`

| Option | Default | Description |
|---|---|---|
| `--max_chunk_size` | `2000` | Maximum characters per chunk |
| `--raw_dir` | `data/raw` | Corpus to index |
| `--index_dir` | `data/processed` | Where the indexes are saved (`index.json`, `vectors.npy`, `owners.npy`, and the embedding model: `embeddings.npy`, `tokenizer.json`) |

```bash
uv run -m src index
uv run -m src index --max_chunk_size 1000
uv run -m src index --raw_dir data/raw
uv run -m src index --index_dir data/processed
uv run -m src index --max_chunk_size 2000 --raw_dir data/raw --index_dir data/processed

make run ARGS="index"
make run ARGS="index --max_chunk_size 1000"
make run ARGS="index --raw_dir data/raw"
make run ARGS="index --index_dir data/processed"
make run ARGS="index --max_chunk_size 2000 --raw_dir data/raw --index_dir data/processed"
```

#### `search`

| Option | Default | Description |
|---|---|---|
| `--query` (or positional) | required | The question |
| `--k` | `10` | Number of sources returned |
| `--index_dir` | `data/processed` | Index built by `index` |
| `--retriever` | `hybrid` | `hybrid` (BM25 re-ranked with the vector index) or `bm25` (words only) |

```bash
uv run -m src search "How do I load a LoRA adapter?"
uv run -m src search --query "How do I load a LoRA adapter?"
uv run -m src search "How do I load a LoRA adapter?" --k 5
uv run -m src search "How do I load a LoRA adapter?" --index_dir data/processed
uv run -m src search --query "How do I load a LoRA adapter?" --k 5 --index_dir data/processed
uv run -m src search "How do I load a LoRA adapter?" --retriever bm25

make run ARGS='search "How do I load a LoRA adapter?"'
make run ARGS='search --query "How do I load a LoRA adapter?"'
make run ARGS='search "How do I load a LoRA adapter?" --k 5'
make run ARGS='search "How do I load a LoRA adapter?" --index_dir data/processed'
make run ARGS='search --query "How do I load a LoRA adapter?" --k 5 --index_dir data/processed'
make run ARGS='search "How do I load a LoRA adapter?" --retriever bm25'
```

#### `search_dataset`

| Option | Default | Description |
|---|---|---|
| `--dataset_path` | required | Dataset of questions (`UnansweredQuestions`) |
| `--k` | `10` | Sources per question |
| `--save_directory` | `data/output/search_results/UnansweredQuestions` | Where the results are written (same file name as the dataset) |
| `--index_dir` | `data/processed` | Index built by `index` |
| `--retriever` | `hybrid` | `hybrid` (BM25 re-ranked with the vector index) or `bm25` (words only) |

```bash
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 5
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --index_dir data/processed
uv run -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --retriever bm25

make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json"
make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 5"
make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results/UnansweredQuestions"
make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --index_dir data/processed"
make run ARGS="search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --retriever bm25"
```

#### `evaluate`

| Option | Default | Description |
|---|---|---|
| `--student_search_results_path` | required | Output of `search_dataset` |
| `--dataset_path` | required | Reference dataset (`AnsweredQuestions`) |
| `--k` | `10` | Largest k for which recall is reported |
| `--show_failed` | `False` | List the questions whose sources were not found |

```bash
uv run -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5
uv run -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --show_failed

make run ARGS="evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json"
make run ARGS="evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5"
make run ARGS="evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --show_failed"
```

#### `answer`

| Option | Default | Description |
|---|---|---|
| `--query` (or positional) | required | The question |
| `--k` | `10` | Sources given to the model as context |
| `--index_dir` | `data/processed` | Index built by `index` |
| `--retriever` | `hybrid` | `hybrid` (BM25 re-ranked with the vector index) or `bm25` (words only) |
| `--dtype` | `auto` | `auto` (chosen from free memory), `float32` or `bfloat16` (forced, no memory check) |

```bash
uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?"
uv run -m src answer --query "What is the minimum version of bitsandbytes required for vLLM quantization?"
uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --k 5
uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --index_dir data/processed
uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --dtype bfloat16
uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --retriever bm25

make run ARGS='answer "What is the minimum version of bitsandbytes required for vLLM quantization?"'
make run ARGS='answer --query "What is the minimum version of bitsandbytes required for vLLM quantization?"'
make run ARGS='answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --k 5'
make run ARGS='answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --index_dir data/processed'
make run ARGS='answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --dtype bfloat16'
make run ARGS='answer "What is the minimum version of bitsandbytes required for vLLM quantization?" --retriever bm25'
```

#### `answer_dataset`

| Option | Default | Description |
|---|---|---|
| `--student_search_results_path` | required | Output of `search_dataset` |
| `--save_directory` | `data/output/search_results_and_answer/UnansweredQuestions` | Where the answers are written |
| `--dtype` | `auto` | As in `answer` |

```bash
uv run -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
uv run -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dtype float32

make run ARGS="answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json"
make run ARGS="answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions"
make run ARGS="answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dtype float32"
```

#### Pipeline rules

`make test_rag` runs on the mini corpus in `tests/data`. It takes these variables:

| Variable | Default | Description |
|---|---|---|
| `TEST_DATASET` | `dataset_mini.json` | Dataset in `tests/data/datasets/*/` |
| `TEST_K` | `5` | k for `search_dataset` and `evaluate` |
| `TEST_CHUNK` | `500` | `--max_chunk_size` for `index` |

```bash
make test_rag
make test_rag TEST_DATASET=dataset_mini_hard.json
make test_rag TEST_K=10
make test_rag TEST_CHUNK=2000
make test_rag TEST_DATASET=dataset_mini_hard.json TEST_CHUNK=2000 TEST_K=10
```

`make test_rag_base` runs on the real data with fixed values (k = 10, chunks of 2000, docs and code datasets). `make cleandata` removes the index and the generated JSON files.

```bash
make test_rag_base
make cleandata
```

Time limits of the subject, on the real data:

| Variable | Default | Description |
|---|---|---|
| `INDEX_LIMIT` | `300` | Seconds allowed for `index` (`test_index`) |
| `RETRIEVAL_LIMIT` | `90` | Seconds allowed for the 200 questions (`test_retrieval`) |

```bash
make test_limits
make test_index INDEX_LIMIT=1        # see a FAIL
```

Answer batteries (they run the model: ~20 s per question):

| Variable | Default | Description |
|---|---|---|
| `ANSWER_ARGS` | empty | Extra options for `answer` (`--k 5 --dtype bfloat16`) |
| `ANSWER_DATASET` | `data/datasets/AnsweredQuestions/dataset_docs_public.json` | Dataset of `test_answer_dataset` |
| `ANSWER_START` | `1` | First question to run |
| `ANSWER_LIMIT` | `0` (all) | Number of questions from there |
| `ANSWER_LOG` | `data/output/answer_logs/<dataset>.log` | Where the output is also saved |

```bash
make test_answer
make test_answer ANSWER_ARGS="--k 5"
make test_answer_dataset
make test_answer_dataset ANSWER_START=77 ANSWER_LIMIT=10
make test_answer_dataset ANSWER_DATASET=data/datasets/AnsweredQuestions/dataset_code_public.json
```

## System architecture

```
                 index
data/raw ──► CorpusReader ──► PyChunker (.py) ──┐   ┌─► BM25Index ─────► save ──► data/processed/index.json
                              MdChunker (.md,   ├───┤   (Tokenizer)
                                        .txt) ──┘   └─► VectorIndex ───► save ──► data/processed/vectors.npy
                              (same chunks, same order)  (windows +             data/processed/owners.npy
                                                          StaticEncoder) ─────► data/processed/embeddings.npy
                                                          potion-base-8M        data/processed/tokenizer.json

                 search / search_dataset / answer            (--retriever hybrid, the default)
question ──► BM25Index.search ──► top 30 candidates ──┐
        └──► VectorIndex.chunk_score ─────────────────┴─► HybridRetriever: re-rank ──► top-k chunks ──► JSON
                                                                                          │
                                                                       answer             ▼
                                                                              Generator (Qwen3-0.6B)
                                                                              _build_context → _build_messages
                                                                              → chat template → generate → answer
```

With `--retriever bm25` the top-k chunks come straight from `BM25Index.search` and the vector index is not loaded. Searching never reads the Hugging Face cache: the embedding model is read from `data/processed`.

| Component | File | Role |
|---|---|---|
| `RagCLI` | `src/commands.py` | The six Fire commands. Each one validates its arguments, calls the components and handles the output. |
| `CliCommand.validated` | `src/commands.py` | Decorator: pydantic `validate_call` on every command (Fire ignores type hints) and a clean signature for `--help`. |
| `ArgvGuard` | `src/argv_guard.py` | Checks `argv` **before** Fire runs: unknown commands and options, private names, misplaced `--help`. Fire would otherwise run the command first and complain afterwards. |
| `CorpusReader` | `src/reader.py` | Reads every `.py`, `.md` and `.txt` file under the corpus folder, with paths relative to the project root (the grader compares paths verbatim). Rejects folders outside the project. |
| `PyChunker`, `MdChunker` | `src/chunker.py` | The two chunking strategies, both built on `FixedSizeChunker` (see below). |
| `Tokenizer` | `src/tokenizer_bm.py` | Text → terms, the same for chunks and queries; drops stopwords from queries only. |
| `BM25Index` | `src/bm25_index.py` | Builds the inverted index, ranks chunks, saves and loads it as JSON. |
| `VectorIndex` | `src/vector_index.py` | Splits every chunk into windows, embeds them with `potion-base-8M`, scores every chunk by its best window; saves and loads two `.npy` files and the model. |
| `StaticEncoder` | `src/static_encoder.py` | The embedding model run with numpy: tokenizer + one vector per token, a text is the normalised mean of its tokens. Loaded from the Hugging Face cache by `index`, then saved with and read from the index. |
| `Retriever` | `src/base_retriever.py` | Abstract base with one method, `search(query, k)`. `BM25Index` and `HybridRetriever` inherit it, so the commands only see a `Retriever`. |
| `HybridRetriever` | `src/hybrid_retriever.py` | Re-ranks the BM25 candidates with the vector scores. A `Retriever` like `BM25Index`; `RagCLI._load_retriever` picks one from `--retriever`. |
| `Generator` | `src/generator.py` | Loads Qwen3-0.6B once, fits the retrieved chunks into the token budget, builds the prompt and generates the answer. Imported only by `answer` / `answer_dataset`, so the other commands never load Qwen. |
| `Evaluator` | `src/evaluator.py` | recall@k with the moulinette's rules (same file, character IoU ≥ 0.05, sources ≤ 2000 characters). |
| Models | `src/models.py` | The subject's pydantic models (`MinimalSource`, `StudentSearchResults`…) plus internal ones (`Chunk`, `Document`, `SearchHit`, `IndexData`, `WindowChunk`). |
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

**Vector re-ranking** (bonuses 1 and 2; `src/vector_index.py`, `src/hybrid_retriever.py`). The default retriever; the choices below are measured in *Semantic embeddings*.

- **Model:** `minishlab/potion-base-8M` (Model2Vec, a static embedding model of ~30 MB). It is only a tokenizer and a table with one 256-number vector per token, so `StaticEncoder` runs it with `tokenizers` + numpy: the mean of the text's token vectors, normalised. Same vectors as `sentence-transformers` (largest difference 9e-8 over the whole corpus) without importing PyTorch. It embeds the corpus in ~4 s on CPU; MiniLM would need ~6 min here and ~20 min on campus.
- **Windows:** each chunk is split into windows of whole lines of up to `MAX_WINDOW_SIZE` (400) characters, and the words of its file path are put in front of each window (`vllm model executor … bitsandbytes py\n<window>`). 59,828 vectors of 256 dimensions, normalised, saved as `vectors.npy` with `owners.npy` (the chunk id of each window).
- **Chunk score:** the question is embedded as written (stopwords kept), the dot product with every window is its cosine similarity, and a chunk keeps the score of its **best** window (`np.maximum.at`).
- **Re-ranking:** BM25 gives its top `CANDIDATES` (30); each gets

  ```
  combined = (1 − ALPHA) × BM25 score / best BM25 score + ALPHA × similarity      (ALPHA = 0.5)
  ```

  and the candidates are sorted again (ties keep the BM25 order). Dividing by the best BM25 score puts both parts on the same 0–1 scale. The vectors only reorder chunks BM25 already found, so they cannot bring in a chunk with none of the question's words.
- **Same chunk ids:** `index` builds both indexes from the same list of chunks, and every `SearchHit` carries its `chunk_id`. `HybridRetriever` checks that both indexes have the same `chunk_count`; if one was rebuilt without the other, it stops with *"Run 'index' again"*.
- **Cost:** ~4 s more for `index` (2.4 s tokenizing, 1.6 s averaging), ~0.3 s to load the model saved with the index, then ~7 ms per question.

**Generation** (`src/generator.py`).
- **Context:** the ranked chunks, best first, each with its file path, until `MAX_CONTEXT_TOKENS` (**2200**, counted with Qwen's own tokenizer).
- **Prompt:**
  - a short system message: answer only from the context, quote the exact names and values, and a fixed refusal sentence when the answer is not there;
  - **two few-shot examples** as earlier turns of the conversation, invented on purpose: one question answered by quoting a value, one unrelated question refused;
  - the real user message: the context first, the **question last**, then a one-line reminder (*"Answer the question using the context above."*).
- **Decoding:** greedy (`do_sample=False`), without Qwen3's thinking mode, at most `MAX_NEW_TOKENS` (150).
- **Clean-up:** if an answer has content and also the refusal sentence, the refusal is removed (it contradicted the answer).
- **Loading:** the model and the tokenizer are loaded from the local cache first (`local_files_only`), so a normal run makes no network request; the type (float32 / bfloat16) is chosen from the free memory (see Performance analysis).

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions) and on the mini datasets in `tests/data` (11 easy, 13 hard questions). `evaluate` gives the same numbers as the moulinette (checked on every run). With ~100 questions, **one question ≈ 1 %**, so differences of a few points can be noise.

### Current results

Moulinette, chunks of 2000 characters, k = 10 (`make test_rag_base`, 2026-10-06):

| Retriever | Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target (@5) |
|---|---|---|---|---|---|---|
| **hybrid** (default) | docs | 0.640 | 0.870 | **0.910** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.545 | 0.778 | **0.828** | 0.859 | ≥ 0.50 ✅ |
| `bm25` | docs | 0.610 | 0.830 | **0.870** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.515 | 0.717 | **0.758** | 0.818 | ≥ 0.50 ✅ |

The hybrid numbers are exactly those of the experiment that chose the configuration (see *Semantic embeddings: tuning*).

| Step | Time | Subject limit |
|---|---|---|
| `index` (1,969 files, 20.1 M characters → 12,907 chunks, both indexes and the model built and saved) | ~13 s (embedding the 59,828 windows: ~4 s) | 5 min |
| `search_dataset`, hybrid (100 questions, including loading both indexes and the embedding model) | ~3 s | 90 s for 200 questions |
| `search_dataset`, `--retriever bm25` | ~2 s | |
| `make test_limits` (hybrid) | index 13.2 s, 199 questions 5.4 s: PASS (with `sentence-transformers`: 28.2 s and 19.4 s) | |
| Loading the BM25 index / the vector index | ~0.8 s / < 0.1 s | |
| Loading the embedding model | ~0.3 s from the index (~6.5 s with `sentence-transformers`, which imports PyTorch) | |
| `answer`: load Qwen3-0.6B | ~5.6 s | |
| `answer`: whole command (load + one answer) | ~15–25 s | no limit for generation |
| `answer_dataset`-like run on the 100 docs questions (`make test_answer_dataset`) | ~37 min (~22 s per question) | |

### Evolution of recall@5

| Date | Change | docs | code |
|---|---|---|---|
| 2026-10-01 | Baseline: fixed-size chunks, plain tokenizer, BM25 `K1 = 1.5`, `B = 0.75` | 0.850 | 0.596 |
| 2026-10-02 | `PyChunker`, file path and snake_case terms, `K1 = 1.9`, `B = 0.5` | 0.830 | 0.727 |
| 2026-10-02 | `MdChunker` with the code-fence separator | 0.860 | 0.727 |
| 2026-10-02 | Dotted-name pass removed, `.txt` with `MdChunker`, decorators kept with their `def` | 0.860 | 0.717 |
| 2026-10-03 | Query stopwords | 0.870 | 0.758 |
| 2026-10-06 | Hybrid retriever: BM25 top 30 re-ranked with `potion-base-8M` windows (α = 0.5) | **0.910** | **0.828** |

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

### Semantic embeddings: choosing the model (bonuses 1 and 2, 2026-10-05)

Before writing the vector index, five lightweight CPU models were measured on the real corpus (12,907 chunks of up to 2000 characters): the time to embed the whole corpus, and recall@k on the public datasets with the vectors alone and fused with BM25 by reciprocal rank fusion (`weight / (60 + rank)` per list). Vectors are L2-normalised, so the similarity is a dot product.

**Speed** (this PC, 8 threads; the campus computers are ~3× slower):

| Model | Kind | Size | Whole corpus | On campus (≈) |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2`, 256 tokens (default) | transformer | 88 MB | ~400 s | ~20 min |
| `all-MiniLM-L6-v2`, cut to 128 tokens | transformer | 88 MB | 344 s | ~17 min |
| `sentence-transformers/static-retrieval-mrl-en-v1` | static (averaged token vectors) | 121 MB | 5.2 s | ~16 s |
| `minishlab/potion-retrieval-32M` (Model2Vec) | static | 125 MB | 2.1 s | ~6 s |
| **`minishlab/potion-base-8M` (Model2Vec)** | static | **30 MB** | **2.0 s** | **~6 s** |

MiniLM does not fit in the 5-minute indexing limit; a static model adds seconds.

**Recall@5** (k = 10; vectors alone, and hybrid with BM25 at equal weight):

| Retrieval | docs @5 | code @5 | docs @10 | code @10 |
|---|---|---|---|---|
| **BM25 (current)** | **0.870** | **0.758** | **0.920** | **0.818** |
| MiniLM 256 tokens, vectors only | 0.510 | 0.374 | 0.580 | 0.455 |
| MiniLM 128 tokens, vectors only | 0.460 | 0.333 | 0.500 | 0.374 |
| **potion-base-8M, vectors only** | 0.440 | 0.343 | 0.520 | 0.414 |
| static-retrieval-mrl, vectors only | 0.350 | 0.293 | 0.420 | 0.374 |
| potion-retrieval-32M, vectors only | 0.340 | 0.263 | 0.390 | 0.354 |
| BM25 + MiniLM 256 (hybrid) | 0.720 | 0.636 | 0.800 | 0.778 |
| BM25 + potion-base-8M (hybrid) | 0.760 | 0.596 | 0.840 | 0.737 |

**Hybrid weights** (BM25 + MiniLM, vector list weighted down):

| Vector weight | MiniLM 128: docs @5 | code @5 | MiniLM 256: docs @5 | code @5 |
|---|---|---|---|---|
| 1 (equal) | 0.700 | 0.576 | 0.720 | 0.636 |
| 0.5 | 0.740 | 0.636 | 0.740 | 0.707 |
| 0.3 | 0.820 | 0.687 | 0.800 | 0.737 |
| 0.5, vectors of the documentation only (`.md` / `.txt`) | 0.820 | 0.747 | 0.830 | 0.707 |

- **With whole chunks, no vector model beats BM25, alone or fused** (fixed by the tuning below). Every fusion lowers recall@5: the vector list brings chunks that do not answer the question into the top 5, and the lower its weight, the less harm it does, but it never adds.
- **Why:** these models are trained on short general English sentences, while the chunks are long pieces of code and technical documentation (a transformer only reads their first 128–256 tokens). The BM25 here is already tuned for this corpus (snake_case terms, file paths, query stopwords), so it leaves little for paraphrase matching to fix. Other projects reporting gains from hybrid retrieval started from a weaker BM25 (~0.82 / 0.55).
- **Model chosen: `minishlab/potion-base-8M`.** It retrieves about as well as MiniLM cut to 128 tokens (0.44 / 0.343 against 0.46 / 0.333), but embeds the whole corpus in ~2 s instead of ~6 min, so the vector index is built inside `index`, far below the 5-minute limit. Loaded by `sentence-transformers` for these experiments, 30 MB (now run with numpy, see *Retrieval method*). Being this fast, it can be tuned cheaply: every variant below takes seconds to measure.
- The retrieval-tuned static models (`static-retrieval-mrl`, `potion-retrieval-32M`) did worse than the general one on these chunks; FastText (averaged word vectors, not trained for search, ~7 GB pre-trained English model) was not tried for the same reasons.

### Semantic embeddings: tuning `potion-base-8M` (2026-10-05)

A static model represents a text by the **average** of its token vectors: over a 2000-character chunk, the few words that answer a question are diluted. Three changes were measured, on their own and together:

- **Windows:** each chunk is split into windows of ~400 characters (whole lines, accumulated until the next line would pass 400), every window gets its own vector, and a chunk scores as its **best** window (max-pooling). 59,778 vectors instead of 12,907.
- **File path:** the words of the chunk's file path (`vllm model executor layers quantization bitsandbytes py`) are prepended to every window, as they were added to the BM25 terms.
- **Query stopwords:** the query is embedded without the BM25 stopwords (`what`, `how`, `the`…).

**Vectors only:**

| Path | Windows | Query stopwords removed | Vectors | Embedding time | docs @5 | docs @10 | code @5 | code @10 |
|---|---|---|---|---|---|---|---|---|
| no | no | no | 12,907 | 2 s | 0.440 | 0.520 | 0.343 | 0.414 |
| no | no | yes | 12,907 | 2 s | 0.410 | 0.520 | 0.313 | 0.424 |
| no | yes | no | 59,778 | 3 s | 0.700 | 0.730 | 0.323 | 0.424 |
| no | yes | yes | 59,778 | 3 s | 0.660 | 0.730 | 0.394 | 0.485 |
| yes | no | no | 12,907 | 2 s | 0.460 | 0.550 | 0.404 | 0.475 |
| yes | no | yes | 12,907 | 2 s | 0.460 | 0.530 | 0.364 | 0.485 |
| **yes** | **yes** | **no** | **59,778** | **3 s** | **0.730** | **0.790** | **0.596** | **0.677** |
| yes | yes | yes | 59,778 | 3 s | 0.720 | 0.790 | 0.576 | 0.636 |

Windows and path together take the vectors from 0.44 / 0.343 to **0.73 / 0.596**; each helps most where the other does not (windows on the docs, the path on the code).

**Combining with BM25.** Two ways, with windows + path vectors (query stopwords removed in this first sweep):

| Method | Parameter | docs @5 | docs @10 | code @5 | code @10 |
|---|---|---|---|---|---|
| BM25 alone | — | 0.870 | 0.920 | 0.758 | 0.818 |
| Reciprocal rank fusion of both lists | vector weight 1.0 | 0.850 | 0.880 | 0.707 | 0.828 |
| | 0.5 | 0.880 | 0.880 | 0.737 | 0.818 |
| | 0.3 | 0.880 | 0.920 | 0.727 | 0.838 |
| | 0.2 | 0.890 | 0.920 | 0.747 | 0.828 |
| | 0.1 | 0.900 | 0.920 | 0.758 | 0.818 |
| **Re-ranking** the BM25 top 20 | α = 0.1 | 0.870 | 0.920 | 0.758 | 0.818 |
| | 0.2 | 0.870 | 0.920 | 0.768 | 0.828 |
| | 0.3 | 0.890 | 0.920 | 0.778 | 0.828 |
| | 0.5 | 0.900 | 0.920 | 0.798 | 0.859 |

**Re-ranking:** BM25 picks the top N candidates, then each gets `(1 − α) × BM25 score / best BM25 score + α × vector similarity` and they are sorted again. The vectors only **reorder** chunks BM25 already found relevant, so they cannot bring unrelated chunks into the top 5, which is what made the list fusion lose.

**Full sweep of the re-ranking**, recall@5 as `docs / code` (BM25 alone: 0.870 / 0.758):

Query embedded as written:

| Candidates N \ α | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| 10 | 0.920 / 0.788 | 0.920 / 0.798 | 0.920 / 0.798 | 0.920 / 0.778 | 0.910 / 0.788 | 0.910 / 0.778 |
| 20 | 0.910 / 0.788 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.798 | 0.910 / 0.808 | 0.910 / 0.778 |
| **30** | 0.910 / 0.798 | 0.910 / 0.818 | **0.910 / 0.828** | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |
| 50 | 0.910 / 0.798 | 0.910 / 0.818 | 0.910 / 0.828 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |

Query without stopwords:

| Candidates N \ α | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| 10 | 0.890 / 0.768 | 0.910 / 0.778 | 0.910 / 0.788 | 0.910 / 0.788 | 0.900 / 0.788 | 0.900 / 0.758 |
| 20 | 0.890 / 0.778 | 0.900 / 0.788 | 0.900 / 0.798 | 0.890 / 0.798 | 0.890 / 0.808 | 0.870 / 0.768 |
| 30 | 0.890 / 0.778 | 0.900 / 0.788 | 0.900 / 0.808 | 0.890 / 0.808 | 0.890 / 0.818 | 0.870 / 0.778 |
| 50 | 0.890 / 0.778 | 0.900 / 0.788 | 0.900 / 0.808 | 0.890 / 0.808 | 0.890 / 0.818 | 0.860 / 0.768 |

**Chosen configuration:** windows + path, query as written, re-ranking of the BM25 top **30** with **α = 0.5**:

| Retrieval | docs @5 | docs @10 | code @5 | code @10 |
|---|---|---|---|---|
| BM25 | 0.870 | 0.920 | 0.758 | 0.818 |
| **BM25 + vector re-ranking** | **0.910** | **0.920** | **0.828** | **0.859** |

- **+4 points on docs and +7 on code** at recall@5, for ~3 s more indexing and milliseconds per query.
- **Not a lucky cell:** every configuration with 20–50 candidates and α between 0.3 and 0.7 beats BM25 (docs 0.89–0.92, code 0.78–0.83). The chosen one is the middle of that plateau, not its best cell.
- **Removing the query stopwords hurts the vectors** (1–2 points), the opposite of BM25: the question words carry part of its meaning for an embedding, while for BM25 they only add noise.
- **Caveat:** the parameters were tuned on the same public datasets they are measured on. The plateau makes overfitting unlikely, but the private datasets of the defense may show a smaller gain.
- **Implemented** in `VectorIndex` and `HybridRetriever` (2026-10-06) with the same results. The implementation keeps windows made only of whitespace (59,828 vectors instead of 59,778); recall is unchanged.

### Generation (2026-10-03)

**Speed.** Qwen3-0.6B is stored in `bfloat16`, and transformers loads it in that type by default. This CPU has no native bfloat16, so it is emulated:

| Model type | Read a ~2000-token prompt | Per generated token |
|---|---|---|
| bfloat16 (default) | 29 s | ~150 ms |
| **float32 (implemented)** | **10 s** | ~150 ms |

Reading the prompt grows with its length: 1,273 tokens take 6 s in float32. The subject sets no time limit for generation (the 90 s are for retrieval).

**Memory-safe loading.** float32 is faster but needs more memory, and when Linux runs out of memory it kills the process with `SIGKILL` (`Killed`, exit code 137): no Python exception, nothing the program can catch. So `Generator` reads `MemAvailable` from `/proc/meminfo` **before** loading the model and chooses the type:

| Available memory | Type | Reason |
|---|---|---|
| ≥ 5.5 GB | float32 | measured peak 5.1 GB + margin |
| ≥ 3.4 GB | bfloat16 | measured peak 3.0 GB + margin |
| < 3.4 GB | none: `RagError`, exit 1 | the model would not fit |
| unknown (no `/proc/meminfo`, not Linux) | bfloat16 | the conservative choice |

The margin covers other programs taking memory while the model loads. Checked end to end with the real CLI (memory reading replaced to simulate each case):

| Case | Result | Exit | Peak RAM | Time |
|---|---|---|---|---|
| This PC (9.4 GB available) | float32, correct answer | 0 | 5.0 GB | 19 s |
| No `/proc/meminfo` | bfloat16, same answer | 0 | 2.6 GB | 42 s |
| 2 GB available | `Not enough memory to load Qwen/Qwen3-0.6B: 2.0 GB available, at least 3.4 GB needed` | 1 | 0.75 GB | 2.6 s |

The model is never loaded when it cannot fit, and no case ends with a traceback.

`--dtype float32` or `--dtype bfloat16` skips the automatic choice (for testing). If the forced type needs more memory than is available, a `WARNING` is printed to stderr, but the model is still loaded: forcing it is the user's decision.

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

The rule fixes out-of-domain questions, but makes the model refuse valid ones: a longer reminder dilutes "quote the exact values", and questions that do not name vLLM ("How do I load a LoRA adapter?") look off-topic. **The rule was dropped.**

**Answer battery.** From here every change was measured on 10 fixed questions (`make test_answer`: 8 from the public datasets, 1 off-topic, 1 trap), scored ✅ 2, 🟡 1, refusal 0, ❌ −1 (wrong and confident):

| Prompt | Score | Confident wrong answers |
|---|---|---|
| Short prompt, with the off-topic rule | 8 / 20 (4 valid questions refused) | 1 |
| Short prompt, no off-topic rule | 7 | 4 |
| + two few-shot examples (one answered, one refused) | 11 | 1 |
| + reminder *"Answer the question using the context above."* instead of *"Answer with the exact values from the context."* (the old wording leaked into the answers: *"The exact values are: …"*) | 16 | **0** |
| + context budget 2200 tokens instead of 2500 | **17** | **0** |

- **Teaching by example instead of a rule:** the refused example made "capital of France" a refusal without refusing valid questions.
- **Sampling** (the model card's values) instead of greedy: more valid questions refused (4 → 6) in the same battery; greedy kept, also because it is reproducible.
- **Context budget** (1500, 2000, 2200, 2500, 2800, 3500 tokens): answers only change when the **number of chunks** that fit changes; more chunks distract the model, fewer drop useful ones. 2200 was the only budget where both the LoRA and the security questions were right, and it gave the best battery score.
- **Thinking mode** (`enable_thinking=True`, tried with a temporary flag): 3–7× slower (65–150 s per answer) and, with the right chunk in the context, still picked the wrong list for the Mixtral question. Discarded.

**Whole docs dataset.** The 100 public docs questions, judged by hand against the reference answers (`make test_answer_dataset`):

| Result | First run | After two fixes |
|---|---|---|
| ✅ correct | 50 | **58** |
| 🟡 partial | 19 | 18 |
| refusal | 20 | **11** |
| ❌ wrong | 11 | 13 |
| Score | 108 / 200 | **121 / 200** |

- The right chunk reached the model in **89 / 100** questions: most of the remaining loss is in generation, not retrieval.
- **Fix 1:** the few-shot example answer ended with `(ServerConfig)` and the model copied it into 4 real answers; the example now has no parenthetical.
- **Fix 2:** 6 answers were right but ended with the refusal sentence; `Generator._clean_answer` removes it.
- Side effect: the new example answer made the model refuse less (20 → 11), with mostly right answers and two new confident mistakes.
- The remaining failures (choosing the wrong list among similar chunks, a source whose own error message contradicts its code) are the "known reasoning limits" of a 0.6B model mentioned in the subject.

### Robustness and performance: how to test it yourself

The memory check reads `/proc/meminfo`. To see each branch work **through the real CLI**, without changing the code and without `sudo`, either use real memory or show the command a different `/proc/meminfo`:

```bash
Q="What is the minimum version of bitsandbytes required for vLLM quantization?"

# 1. Real memory pressure: hold most of the free memory in another process
avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
python3 -c "import time; x = b'x' * (($avail - 5) * 1024**3); time.sleep(120)" &
sleep 5; uv run -m src answer "$Q"          # → bfloat16 (~5 GB left)
kill %1

# 2. Fake values: a private mount namespace (unshare -rm, no sudo) where only
#    this command sees another /proc/meminfo
printf 'MemAvailable: 2097152 kB\n' > /tmp/meminfo_2gb      # 2 GB
printf 'MemAvailable: 4194304 kB\n' > /tmp/meminfo_4gb      # 4 GB
unshare -rm sh -c "mount --bind /tmp/meminfo_2gb /proc/meminfo && uv run -m src answer '$Q'"
unshare -rm sh -c "mount --bind /tmp/meminfo_4gb /proc/meminfo && uv run -m src answer '$Q'"

# 3. Unreadable memory: /proc/meminfo replaced by an empty file
unshare -rm sh -c "mount --bind /dev/null /proc/meminfo && uv run -m src answer '$Q'"

# 4. Forced type with too little memory: warning, then it loads anyway
unshare -rm sh -c "mount --bind /tmp/meminfo_2gb /proc/meminfo && uv run -m src answer '$Q' --dtype float32"
```

Results on this PC (16 GB, 12 GB available):

| Test | Model type | Output | Exit | Time | Peak RAM |
|---|---|---|---|---|---|
| No pressure | float32 | 0.46.1 ✅ | 0 | 19 s | 5.0 GB |
| 1. 7 GB held by another process (5.1 GB left) | bfloat16 | 0.46.1 ✅ | 0 | 43 s | |
| 2. Fake 4 GB | bfloat16 | 0.46.1 ✅ | 0 | 45 s | 2.6 GB |
| 2. Fake 2 GB | not loaded | `Not enough memory to load Qwen/Qwen3-0.6B: 2.00 GB available, at least 3.4 GB needed` | 1 | 2.7 s | 0.75 GB |
| 3. Empty `/proc/meminfo` | bfloat16 | `The available memory cannot be read. Torch mode: bfloat16`, then 0.46.1 ✅ | 0 | 44 s | 2.5 GB |
| 4. `--dtype float32`, fake 2 GB | float32 | `WARNING: '--dtype float32' needs at least 5.5 GB of available memory, but only 2.00 GB is available.`, then 0.46.1 ✅ | 0 | | |

`unshare -rm` needs unprivileged user namespaces. Ubuntu 24.04 blocks them by default (`kernel.apparmor_restrict_unprivileged_userns = 1`), and a normal terminal fails with `unshare: write error /proc/self/uid_map: Operation not permitted`. They work from VS Code's integrated terminal when VS Code is the snap package (its AppArmor profile is in complain mode), or after `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` (restore it with `=1` afterwards; it also goes back to 1 on reboot). Test 1 needs neither.

**Known limitation.** `/proc/meminfo` shows the memory of the whole machine. Inside a container or a cgroup with its own limit (`docker --memory`, `systemd-run -p MemoryMax=`), the check sees the host's memory, can choose float32 and the process can still be killed. Reading the cgroup limit (`/sys/fs/cgroup/memory.max`) would fix it; not needed on the 42 computers.

### Robustness and performance improvements

What was tried to make the system faster and harder to break, with its measured effect:

| Attempt | Problem | Result | Kept |
|---|---|---|---|
| Own BM25 with a JSON index (pydantic) | Indexing and search time | Index ~2 s (limit 5 min), 100 questions < 1 s (limit 90 s for 200) | ✅ |
| Dotted-name tokenizer pass | Recall of quoted module names | No gain, more terms to index | ❌ removed |
| Chunk overlap | Answers split between chunks | Hurts docs, gains on code are noise | ❌ |
| Lazy import of `Generator` | PyTorch takes seconds to load | `index`, `search`, `evaluate`, `--help` never load PyTorch | ✅ |
| Load the model once | `answer_dataset` with many questions | One load (~5.6 s) per command, not per question | ✅ |
| `float32` instead of the default `bfloat16` | bfloat16 is emulated on this CPU | Prompt read 29 s → 10 s; answer ~40 s → ~19 s | ✅ when it fits |
| Memory check before loading | Out of memory → `Killed`, nothing to catch | float32, bfloat16 or a clean error, from the available memory | ✅ |
| `--dtype` with a closed list (`Literal`) | Any text was accepted | `float16` → clean argument error, values shown in `--help` | ✅ |
| Shorter prompt asking for exact values | Evasive answers with the right context | Fixed the Marlin question, no regression | ✅ |
| Off-topic rule in the reminder | "Paris" answered from the model's own knowledge | Refuses out-of-domain questions, but also some valid ones | ❌ replaced by a few-shot refusal example |
| Bugs found by running the tests above | — | `"MemAvaiable"` typo (never read the memory) and `dtype` passed positionally (`TypeError`: the model never loaded) | ✅ fixed |
| Cache-first model loading (`local_files_only`) | `from_pretrained` asks the Hugging Face Hub for updates on every run; one request never answered and a run hung for minutes | No network request when the model is cached; works offline (checked with no network at all); a failed first download is a clean error | ✅ |
| Source paths checked in `answer_dataset` | A search results JSON could point at `/etc/passwd` or `../../.ssh/id_rsa`, which would go into the prompt | Every path is normalised and must stay inside the project; checked before PyTorch is loaded | ✅ |
| Paths normalised without following symbolic links | A `data/raw` linked to another disk (`goinfre` on 42 computers) was rejected as "outside the project" | Linked corpus accepted, stored paths keep the `data/raw/...` form, `--raw_dir ../` still rejected | ✅ |
| No model for empty inputs | `answer` with no matching source, `answer_dataset` with no question | The model is not loaded: fixed message or empty output | ✅ |
| Static embedding model (Model2Vec `potion-base-8M`) instead of MiniLM | MiniLM needs ~6 min on this PC (~20 min on campus) to embed the corpus | ~2 s, about the same recall as MiniLM at 128 tokens; see *Semantic embeddings* | ✅ |
| Windows + file path for the vectors, re-ranking of the BM25 top 30 | Averaged vectors of whole chunks lowered recall when fused with BM25 | Recall@5 docs 0.870 → 0.910, code 0.758 → 0.828 (same as the experiment) | ✅ default, `--retriever bm25` to compare |
| Vector index saved as `.npy`, checked by hand on load | No pydantic for arrays; a missing or foreign file must not crash | Missing file → *"Run 'index' first"*; garbage or arrays that do not fit → clean `VectorIndexError` | ✅ |
| Embedding model run with numpy (`StaticEncoder`) and saved with the index (bonus 4) | `sentence-transformers` imported PyTorch: ~6.5 s per command before the first query | Same vectors (max difference 9e-8), same recall; model loads in ~0.3 s; `index` 28.2 s → 13.2 s, 199 questions 19.4 s → 5.4 s; no PyTorch and no Hugging Face cache when searching | ✅ |
| Averaging the token vectors without a Python loop (`np.add.reduceat`, a count matrix times the table) | The loop over 59,828 windows looked slow | Loop 1.6 s; `reduceat` 13–16 s (copies 7.3 M token rows before adding), count matrix 1.6 s: the work is in the additions, not in Python | ❌ loop kept |

### Tests

`make test` runs **1,116 tests** in ~60 s, without loading Qwen or the embedding model (the memory-safe loading and the answers are checked end to end as described above):

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
- **Vector index and hybrid retriever** (`tests/conftest.py` replaces the embedding model, for every test, with a fake bag-of-words encoder):
  - windows (whole lines, the long-line case, joined back to the text), path prefix, owners in chunk order;
  - best window per chunk (the worked example of the plan), scores in [−1, 1];
  - save/load round trip, missing file, garbage file, arrays that do not fit;
  - the combination formula on fixed scores, `ALPHA = 0` gives exactly the BM25 order, `ALPHA = 1` the similarity order, only BM25 candidates come out, ties keep the BM25 order, mismatched indexes rejected.
- **`StaticEncoder`** with a tiny real model built in the test (a 5-word tokenizer and a table of known vectors): mean and normalisation computed by hand, unknown words, empty texts (zeros, never NaN), save/load round trip, missing or broken files (a broken tokenizer gives a clean error, not a Rust traceback), and `from_pretrained` from the cache, with a download, and with neither. `Retriever` cannot be instantiated and both retrievers inherit it.
- **Evaluator:** IoU, cutoffs, missing and unknown questions, validation errors, the same numbers as the moulinette.
- **CLI:** every command, argument validation and error messages, unknown or private options, exit codes, `--help`.
- **Paths:** corpus folders outside the project rejected, symbolic links accepted, source paths of `answer_dataset` checked (absolute paths and `..`).
- **Generator** (without the model): the refusal clean-up and the few-shot example.
- **Mini datasets** in `tests/data`: a small vLLM-like corpus with easy and hard questions, also run end to end by `make test_rag`.

## Design decisions

- **BM25 written from scratch** instead of a library: the inverted index, idf, saturation and tie-breaking are explicit and testable against the formula, and the index format is our own pydantic model.
- **JSON index validated with pydantic** instead of pickle. It loads in ~0.35 s, can be inspected, and a corrupted or foreign file gives a clear error instead of arbitrary behaviour.
- **Two chunkers sharing one cutting algorithm.** `PyChunker` and `MdChunker` only declare their separators. The invariants are proven once, for every strategy.
- **Measure every change on its own.** Each idea (path terms, tokenizer passes, heading cuts, overlap, stopwords) was measured alone against the previous configuration, and kept only if it helped. Heading cuts and overlap sounded right and were discarded by the numbers.
- **Vectors re-rank BM25 instead of fusing two lists.** Fused lists let the vectors bring in chunks with none of the question's words, and lost to BM25 alone; re-ranking the BM25 top 30 only reorders good candidates (see *Semantic embeddings*).
- **Three classes for the hybrid search:** `BM25Index` searches by words, `VectorIndex` scores by meaning, `HybridRetriever` combines them. `VectorIndex` does not depend on BM25. Both retrievers inherit the abstract `Retriever` (one method, `search`), so the commands choose one in a single place (`_load_retriever`) and only see a `Retriever`; a cache can wrap any of them.
- **A static model needs no deep-learning library.** `potion-base-8M` is a lookup table: running it with numpy gives the same vectors without PyTorch. The model is saved inside the index folder, so a search depends only on `data/processed` (no network, no Hugging Face cache).
- **A static embedding model with windows** instead of a transformer like MiniLM: it fits the campus time limit with room to spare, and windows plus max-pooling recover what averaging a whole chunk loses.
- **Stopwords on the query only.** The index keeps every term, so the list can change without rebuilding it.
- **Lazy import of the generator.** `answer` and `answer_dataset` import `Generator` inside the command, so `index`, `search`, `evaluate` and `--help` never pay the seconds PyTorch takes to load.
- **Load the model once.** `Generator.__init__` loads Qwen; each question only calls `generate`.
- **float32 on CPU when it fits**: 3× faster to read the prompt than the default bfloat16, at the cost of more RAM (peak ~5.1 GB against ~3.0 GB). The type is chosen from the available memory before loading, so a smaller machine gets a slower answer instead of a killed process.
- **Prompt for a small model:**
  - a short system message with positive instructions;
  - the question after the context, where a small model attends most;
  - the key rule repeated right before the answer;
  - a fixed refusal sentence, easy to detect;
  - few-shot examples instead of extra rules: a 0.6B model follows an example better than an instruction;
  - greedy decoding, so answers are reproducible.
- **A fixed answer battery for every prompt change**, scored by hand, then the whole docs dataset: prompt choices are measured like retrieval choices.
- **No network at run time:** the model is loaded from the local cache; the Hub is only used for the first download.
- **Never trust paths that come from a file:** the source paths of a results JSON are checked against the project root before anything is read.
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
- **Running out of memory cannot be caught.** The kernel kills the process without a Python exception. Solved by checking the available memory first: float32, bfloat16 or a clean error, depending on what fits.
- **Evasive answers with a correct context.** Diagnosed with the oracle test (only the correct chunk) and solved with a shorter prompt that asks for the exact values.
- **Refusing out-of-domain questions without refusing valid ones.** The 0.6B model answered "Paris" from its own knowledge despite the rules. An explicit off-topic rule fixed it but also made it refuse valid questions. Solved with a few-shot refusal example instead of the rule (battery score 7 → 17, no confident wrong answer).
- **The prompt leaking into the answers.** The reminder *"Answer with the exact values"* came back as *"The exact values are: …"*, and the example's `(ServerConfig)` was copied into real answers. Found by reading the outputs, fixed by rewording the reminder and the example.
- **A run that hung with no error.** The process was asleep on an HTTPS connection to the Hugging Face Hub that never answered (seen with `ss`). Solved by loading from the local cache first.

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

- **Learning.** Explanations of the subject, RAG, BM25, tokenization, embeddings, PyTorch and transformers, with exercises solved step by step (personal notes, not part of the repository).
- **Review.** Reviews of my code (bugs found in the reader, the index, the evaluator and the CLI), of my design choices and of the Fire edge cases.
- **Code, on request:**
  - the CLI hardening (`CliCommand.validated`);
  - The decorator handling of `PyChunker`;
  - the query stopword list;
  - most of the test suite.
- **Experiments.** Scripts for the ablations and measurements reported above (chunk sizes, path terms, Markdown variants, overlap, stopwords, model types, prompt variants), and the analysis of their results.
- **Documentation.** Drafts of the docstrings and of this README.

Every part was reviewed, run and understood before being kept, and the design decisions were mine.