*This project has been created as part of the 42 curriculum by saperez-.*

# RAG against the machine

> A Retrieval-Augmented Generation system that answers questions about the vLLM codebase. It runs entirely on CPU, with a BM25 search engine written from scratch, a semantic re-ranker and a small local language model (Qwen3-0.6B).

| Retrieval (recall@5, moulinette) | Indexing | Retrieval speed | Generation |
|---|---|---|---|
| **docs 91.0 %** (target 80 %) · **code 82.8 %** (target 50 %) | ~13 s for 1,969 files (limit 5 min) | 199 questions in ~5.4 s (limit 90 s for 200), ~0.7 s when cached | ~17–22 s per answer on CPU · 76 % correct or partial on the docs dataset |

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

A language model only knows what it was trained on. When you ask a small model about a specific codebase, it either gives no useful answer or makes up a confident one. **Retrieval-Augmented Generation (RAG)** avoids retraining the model. When a question arrives, the system finds the relevant parts of the codebase, gives them to the model as context, and tells it to answer only from that context.

This project applies RAG to the [vLLM](https://github.com/vllm-project/vllm) repository, version 0.10.1: 1,969 Python, Markdown and text files, about 20 million characters. The pipeline has four stages:

1. **Indexing.** Every file is split into chunks of at most 2,000 characters, with one strategy for Python code and another for Markdown and text. The chunks are stored in a lexical index (BM25) and a vector index (embeddings).
2. **Retrieval.** BM25 matches the question against the index, and its best candidates are **re-ranked by meaning** with a lightweight static embedding model (`minishlab/potion-base-8M`). The result is a ranked list of source locations, each one a file path and a character range.
3. **Augmentation.** The best chunks are placed in the prompt, up to a fixed token budget.
4. **Generation.** **Qwen/Qwen3-0.6B** writes the answer from that context, or says that the context does not contain it.

The system is judged on two things:

- **Retrieval quality**, measured with recall@k against reference datasets. The targets are recall@5 ≥ 80 % on documentation questions and ≥ 50 % on code questions.
- **Answer quality**: answers must be coherent, on point and grounded in the retrieved sources.

Everything is exposed through a command-line interface built with Python Fire: `index`, `search`, `search_dataset`, `answer`, `answer_dataset` and `evaluate`.

### Project status

| Part | Status |
|---|---|
| Mandatory part | ✅ Complete: the six commands work end to end, and every target is met by both retrievers |
| Bonus 1: semantic embeddings | ✅ Vector index built with `potion-base-8M` next to the BM25 index |
| Bonus 2: hybrid retrieval | ✅ BM25 candidates re-ranked with the vector scores (default; `--retriever bm25` gives the lexical search alone) |
| Bonus 4: caching | ✅ Cold start: the embedding model is saved with the index and run with numpy, so searching never imports PyTorch. Repeated queries: results are cached on disk, so a repeated search loads no index at all (`--cache=False` to turn it off) |
| Bonus 3: incremental indexing · Bonus 5: local HTTP API | Not started |

---

## Instructions

### Requirements

| Requirement | Details |
|---|---|
| Python | ≥ 3.10 |
| Package manager | [uv](https://docs.astral.sh/uv/). It is the only tool you need: `uv sync` installs everything. |
| Hardware | CPU only. PyTorch comes from the CPU wheel index configured in `pyproject.toml`, so no CUDA libraries are downloaded. |
| Memory for retrieval | ~0.5 GB (`index`, `search`, `search_dataset`, `evaluate`) |
| Memory for generation | The model type is chosen from the available memory before loading. It uses **float32** (fastest) when at least 3.5 GB is available and **bfloat16** (~2–3× slower on CPU) when at least 2.0 GB is available. With less, `answer` stops with a clear error before loading anything. |
| Network | Only the first time: `index` downloads the embedding model (~30 MB) and the first `answer` downloads Qwen3-0.6B (~1.2 GB) to the Hugging Face cache. After that, every run works offline. |

### Installation

```bash
uv sync            # or: make install
```

`uv sync --locked` installs exactly the versions in `uv.lock`, and fails if the lock file is out of date.

On computers with a small home directory (for example at 42), the virtual environment and the model cache can live elsewhere. `make` detects `/sgoinfre/students/$USER` and exports these variables automatically. You can also set them yourself:

```bash
export UV_PROJECT_ENVIRONMENT=/sgoinfre/students/$USER/rag-venv
export HF_HOME=/sgoinfre/students/$USER/hf
```

### Project layout

```
.
├── src/                 # the Python package, run with: uv run python -m src <command>
│   ├── __main__.py      # entry point: Fire, error handling and exit codes
│   ├── commands.py      # RagCLI: the six commands
│   ├── argv_guard.py    # command-line checks before Fire runs
│   ├── reader.py        # CorpusReader
│   ├── chunker.py       # FixedSizeChunker, PyChunker, MdChunker
│   ├── tokenizer_bm.py  # Tokenizer (BM25 terms, query stopwords)
│   ├── base_retriever.py, bm25_index.py, vector_index.py,
│   │   static_encoder.py, hybrid_retriever.py, cached_retriever.py
│   ├── generator.py     # Generator (Qwen3-0.6B)
│   ├── evaluator.py     # recall@k, same rules as the moulinette
│   ├── models.py        # pydantic models
│   └── exceptions.py    # RagError hierarchy
├── tests/               # pytest suite, mini corpus and answer battery
├── Makefile
├── pyproject.toml
└── uv.lock
```

The corpus, the datasets, the index and the outputs are not part of the repository. They live in the folders the subject defines (raw corpus, `processed` index, `UnansweredQuestions` / `AnsweredQuestions` datasets, and outputs scoped by dataset). Every path is a CLI option, and the defaults follow that layout.

### Running

```bash
uv run python -m src <command> [options]     # or: uv run -m src ...
uv run -m src <command> --help               # options of a command
```

### Command reference

| Command | Options (default) | Output |
|---|---|---|
| `index` | `--max_chunk_size` (2000, at most 2000) · `--raw_dir` (`data/raw`) · `--index_dir` (`data/processed`) | The BM25 index (`index.json`), the vector index (`vectors.npy`, `owners.npy`) and the embedding model (`embeddings.npy`, `tokenizer.json`). Removes the query caches of the previous index |
| `search <query>` | `--k` (10) · `--index_dir` · `--retriever` (`hybrid` \| `bm25`) · `--cache` (`True`) | `StudentSearchResults` JSON on stdout |
| `search_dataset` | `--dataset_path` (required) · `--k` (10) · `--save_directory` (`data/output/search_results/UnansweredQuestions`) · `--index_dir` · `--retriever` · `--cache` | `StudentSearchResults` JSON, with the same file name as the dataset |
| `answer <query>` | `--k` (10) · `--index_dir` · `--retriever` · `--dtype` (`auto` \| `float32` \| `bfloat16`) · `--cache` | `StudentSearchResultsAndAnswer` JSON on stdout |
| `answer_dataset` | `--student_search_results_path` (required) · `--save_directory` (`data/output/search_results_and_answer/UnansweredQuestions`) · `--dtype` | `StudentSearchResultsAndAnswer` JSON, with the same file name as the input |
| `evaluate` | `--student_search_results_path` (required) · `--dataset_path` (required, an `AnsweredQuestions` file) · `--k` (10) · `--show_failed` | recall@1/3/5/10 report |

- Progress bars and status messages go to stderr, so you can redirect or pipe the JSON printed by `search` and `answer` (`> out.json`, `| jq`).
- Put boolean flags such as `--show_failed` after the paths, or write them as `--show_failed=True`. Otherwise Fire reads the next word as the flag's value.
- `--dtype` and `--retriever` only accept the listed values. Any other value is rejected with a clear argument error.
- `--cache` is on by default: searches already made with the same index and retriever are read from `<index_dir>/query_cache_<retriever>.json`. Turn it off with `--cache=False`.

### Makefile

| Rule | What it does |
|---|---|
| `make install` | `uv sync` |
| `make run ARGS="..."` | `uv run -m src $(ARGS)` |
| `make debug ARGS="..."` | The same, under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the subject's flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make test` | The pytest suite |
| `make test_rag` | index → search_dataset → evaluate on the mini corpus in `tests/data` (variables `TEST_DATASET`, `TEST_K`, `TEST_CHUNK`) |
| `make test_rag_base` | The same on the real data, as the moulinette runs it: k = 10, chunks of 2000, docs and code datasets |
| `make test_limits` | Checks the subject's time limits and prints PASS or FAIL: indexing ≤ 5 min (`test_index`) and 200 questions ≤ 90 s (`test_retrieval`) |
| `make test_limits_cached` | `test_limits` (indexing removes the query cache, so the searches are uncached), then `test_retrieval` again without indexing, with every question read from the cache |
| `make test_answer` | Runs `answer` on the 10-question battery in `tests/answer_battery` and prints the expected answer after each one |
| `make test_answer_dataset` | The same on a whole `AnsweredQuestions` dataset, with a log (variables `ANSWER_DATASET`, `ANSWER_START`, `ANSWER_LIMIT`, `ANSWER_ARGS`) |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and the `test_rag` outputs |
| `make cleandata` | Removes the generated index and JSON outputs, and keeps the folder layout |

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

**3. Score the results with the moulinette.**

```console
$ ./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.640 Recall@3: 0.870 Recall@5: 0.910 Recall@10: 0.920
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

Each entry of the output keeps the `question_id`, the question and the retrieved sources from step 2, and adds the answer:

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
$ uv run -m src search "How do I load a LoRA adapter?" --k 5 --retriever bm25
```

Each command prints a `StudentSearchResults` JSON with the 5 best sources: file path and character range. Run the same search again and it is read from the query cache, without loading any index. `--cache=False` searches again:

```console
$ uv run -m src search "How do I load a LoRA adapter?" --k 5 --cache=False
```

```console
$ uv run -m src answer "What is the minimum version of bitsandbytes required for vLLM quantization?"
```

This prints a `StudentSearchResultsAndAnswer` JSON whose answer is *"The minimum version of bitsandbytes required for vLLM quantization is 0.46.1."*

- A question about a flag that does not exist (`--turbo-mode`), or one unrelated to the codebase ("What is the capital of France?"), gets the fixed refusal: *"The context does not contain the answer."*
- A query that matches nothing (`"???"`, or only stopwords) gets *"No relevant sources were found in the corpus."* without loading the model.

### Own evaluation

`evaluate` computes the same recall as the moulinette, so you can iterate without it. `--show_failed` lists the questions whose sources were not found:

```console
$ uv run -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json \
    --show_failed
```

### With `make`

The whole command goes inside `ARGS`. If the question has spaces, use single quotes outside so the inner double quotes reach the program:

```bash
make run ARGS='search "How do I load a LoRA adapter?" --k 5'
make test_rag TEST_DATASET=dataset_mini_hard.json TEST_K=10
make test_answer_dataset ANSWER_START=77 ANSWER_LIMIT=10
```

---

## System architecture

```
                 index
data/raw ──► CorpusReader ──► PyChunker (.py) ──┐   ┌─► BM25Index ──────────► index.json
                              MdChunker (.md,   ├───┤   (Tokenizer)
                                        .txt) ──┘   └─► VectorIndex ────────► vectors.npy, owners.npy
                              (same chunks, same order)  windows + StaticEncoder ► embeddings.npy, tokenizer.json

                 search / search_dataset / answer                (--retriever hybrid, the default)
question ──► BM25Index.search ──► top 30 candidates ──┐
        └──► VectorIndex.chunk_score ─────────────────┴─► HybridRetriever: re-rank ──► top-k sources ──► JSON
                                                                                          │
                                                          answer / answer_dataset         ▼
                                                              Generator (Qwen3-0.6B, loaded once)
                                                              context within the token budget → prompt
                                                              → greedy generation → answer → JSON
```

With `--retriever bm25`, the top-k sources come straight from `BM25Index.search` and the vector index is never loaded. Searching never reads the Hugging Face cache, because the embedding model is read from the index folder.

With `--cache` (the default), the chosen retriever is wrapped in a `CachedRetriever`. A question already searched is answered from `query_cache_<retriever>.json`, and the indexes are only loaded by the first question that is not cached, so a fully cached command never loads them.

| Component | File | Role |
|---|---|---|
| `RagCLI` | `src/commands.py` | The six Fire commands. Each one validates its arguments, calls the components and writes the output. `_load_retriever` is the only place where a retriever is chosen. |
| `CliCommand.validated` | `src/commands.py` | Decorator that runs pydantic `validate_call` on every command (Fire ignores type hints) and gives `--help` a clean signature. |
| `ArgvGuard` | `src/argv_guard.py` | Checks the command line **before** Fire runs it: unknown commands and options, private names, extra arguments, a misplaced `--help`. Without it, Fire would run the command first and complain afterwards. |
| `CorpusReader` | `src/reader.py` | Reads every `.py`, `.md` and `.txt` file under the corpus folder. Paths are stored relative to the project root, because the grader compares them verbatim. Folders outside the project are rejected. |
| `PyChunker`, `MdChunker` | `src/chunker.py` | The two chunking strategies, built on a shared `FixedSizeChunker`. |
| `Tokenizer` | `src/tokenizer_bm.py` | Turns text into terms, the same way for chunks and queries. Stopwords are removed from queries only. |
| `Retriever` | `src/base_retriever.py` | Abstract base with one method, `search(query, k)`. The commands only see a `Retriever`. |
| `BM25Index` | `src/bm25_index.py` | A `Retriever`. Builds the inverted index, ranks chunks, and saves and loads the index as JSON. |
| `VectorIndex` | `src/vector_index.py` | Splits every chunk into windows, embeds them and scores every chunk by its best window. Saves and loads two `.npy` files and the model. |
| `StaticEncoder` | `src/static_encoder.py` | The embedding model run with numpy: a tokenizer plus one vector per token. A text's vector is the normalised mean of its token vectors. |
| `HybridRetriever` | `src/hybrid_retriever.py` | A `Retriever` that re-ranks the BM25 candidates with the vector scores. |
| `CachedRetriever` | `src/cached_retriever.py` | A `Retriever` that wraps another one (decorator pattern), remembers its results by question and `k`, saves them to disk between commands, and only loads the wrapped retriever when a question is not cached. |
| `Generator` | `src/generator.py` | Loads Qwen3-0.6B once, fits the chunks into the token budget, builds the prompt and generates the answer. |
| `Evaluator` | `src/evaluator.py` | recall@k with the moulinette's rules: same file, character IoU ≥ 0.05, sources of at most 2,000 characters. |
| Models | `src/models.py` | The subject's pydantic models (`MinimalSource`, `StudentSearchResults`…) and internal ones (`Chunk`, `Document`, `SearchHit`, `IndexData`, `WindowChunk`, `QueryCache`). |
| Errors | `src/exceptions.py` | One hierarchy (`RagError`). Every expected failure ends with a clear message and an exit code, never a traceback. |

- `answer` and `answer_dataset` import `Generator` inside the command, so `index`, `search`, `search_dataset`, `evaluate` and `--help` never load PyTorch.
- `answer_dataset` reads the output of `search_dataset`, which only stores the location of each source. It rebuilds each chunk's text from its file and character range, reading every file once. Every source path is checked first: a path outside the project (`/etc/passwd`, `../..`) is rejected before PyTorch is loaded.

---

## Chunking strategy

Every file is split into chunks of at most `--max_chunk_size` characters. The default is 2,000, the moulinette's limit, and larger values are rejected. Offsets are exact: `text[first_character_index:last_character_index]` (end exclusive) gives back the chunk. Chunks never overlap, and only whitespace is left out. The corpus gives **12,907 chunks**.

### Common mechanism: `FixedSizeChunker`

1. The window `[start, start + size)` is cut at the best separator in a priority list, found with `rfind`.
2. A separator is only accepted **in the second half of the window**. This minimum fill avoids tiny chunks.
3. If no separator is found, the cut is hard.

The two strategies only change the separator list and `CUT_BEFORE`, the separators that start the next chunk instead of ending the current one. That way the size, offset and progress invariants are written and tested once, for both strategies.

### Python code: `PyChunker`

- **Separators, best first:** a top-level decorator, `class`, `def`, `async def`; then the same separators indented by four spaces (methods); then a blank line, a newline and a space.
- **Cut before definitions.** The cut goes before the keyword, so each chunk starts with a whole class, function or method when it fits. 51 % of its chunks start at a definition, against 27 % with plain fixed-size cuts.
- **Decorators stay with their definition.** A cut that lands on a definition is moved back over the `@decorator` lines just above it. Chunks that end with a decorator line went from 406 to 1. The cut never moves before the start of the chunk, so chunking always advances.

### Markdown and text: `MdChunker`

- **Separators, best first:** the end of a fenced code block (`\n```\n`), then a blank line, a newline and a space.
- **Code blocks stay whole.** In the vLLM docs almost every opening fence names a language (```` ```bash ````, ```` ```python ````): 820 such openings against 2 bare ones. So `\n```\n` matches the closing fences, and a command or snippet stays whole in the chunk that closes it.
- **Headings are not separators.** 79 of the 100 docs reference sources start at a heading, yet every heading-based variant measured about 4 points lower (see [Performance analysis](#markdown-chunking-variants)).
- `.txt` files use the same strategy, which is the subject's "Markdown / text" chunking.

---

## Retrieval method

Retrieval has two layers. **BM25**, implemented from scratch, finds the candidates. A **semantic re-ranker** then reorders them. `--retriever bm25` runs the first layer alone.

### Tokenizer

The text is lowercased and read in two passes:

1. Every run of letters or digits: `max_model_len` → `max`, `model`, `len`.
2. snake_case names kept whole, as an extra term: `max_model_len`. A question that quotes an identifier matches it exactly, and the parts still match a question that only names one of them.

The **file path** of each chunk is tokenized too, and its terms are added to the chunk's terms. 35 of the 99 code questions contain words (`keye`, `sampler`, `pynccl`) that only appear in the path of the right file.

### BM25 index and ranking

The index is an inverted index `term → {chunk id: term frequency}`, plus the length of each chunk in tokens. `index` builds it once, saves it as JSON, and pydantic (`IndexData`) validates it on load.

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

Only chunks that share at least one term with the query are scored. The top k are taken with `heapq.nsmallest` on `(−score, chunk id)`. The id breaks ties, so the ranking never depends on set iteration order and is identical between runs.

**Query stopwords.** Question words and common English words (`what`, `how`, `the`, `use`…) are removed from the **query only**. These words are rare in a code corpus, so BM25 would give them a high idf (`why` 5.54, `how` 3.78, against 2.45 for `lora`), and chunks that only share the wording of the question would rank high. The index keeps every term, so the list can change without rebuilding it.

### Semantic re-ranking (bonuses 1 and 2)

- **Model.** `minishlab/potion-base-8M` is a Model2Vec static embedding model of ~30 MB. It is only a tokenizer and a table with one 256-dimension vector per token, so `StaticEncoder` runs it with `tokenizers` and numpy: a text's vector is the mean of its token vectors, normalised. It gives the same vectors as `sentence-transformers` (largest difference 9e-8 over the whole corpus) without importing PyTorch.
- **Windows.** Each chunk is split into windows of whole lines of up to 400 characters, and the words of the chunk's file path are put in front of each window (`vllm model executor … bitsandbytes py\n<window>`). This gives 59,828 normalised vectors, saved as `vectors.npy`, plus `owners.npy` with the chunk id of each window.
- **Chunk score.** The question is embedded as written, with its stopwords. The dot product with every window is the window's cosine similarity, and each chunk keeps the score of its **best** window (`np.maximum.at`).
- **Re-ranking.** BM25 returns its top `CANDIDATES` (30), and each candidate gets a new score:

  ```
  combined = (1 − ALPHA) × BM25 score / best BM25 score + ALPHA × similarity      (ALPHA = 0.5)
  ```

  The candidates are then sorted again, and ties keep the BM25 order. Dividing by the best BM25 score puts both parts on the same 0–1 scale. The vectors only reorder chunks BM25 already found, so they can never bring in a chunk that shares no word with the question.
- **Consistency.** `index` builds both indexes from the same list of chunks, and every `SearchHit` carries its `chunk_id`. `HybridRetriever` checks that both indexes have the same number of chunks. If one was rebuilt without the other, it stops with *"Run 'index' again"*.

---

## Answer generation

### Context

The retrieved chunks are added best first. Each one is a block with a `[CHUNK i]` header, its file path and its text. Chunks are added until the budget of **2,200 tokens** is reached (`MAX_CONTEXT_TOKENS`, counted with Qwen's own tokenizer). The first chunk that does not fit stops the loop, so the best chunks are always kept.

### Prompt

The prompt is designed for a 0.6B model, which follows examples better than long lists of rules.

| Part | Content |
|---|---|
| System message | Answer only from the context and quote the exact names and values. When the answer is not there, give the fixed sentence *"The context does not contain the answer."* |
| Two few-shot turns | One invented question answered by quoting the context, and one unrelated question refused. The examples are invented on purpose, so the model learns the behaviour and not the content. |
| User message | The context first, then the **question last**, where a small model pays the most attention, then a one-line reminder: *"Answer the question using the context above."* |

### Decoding and post-processing

- **Greedy decoding** (`do_sample=False`). The same input always gives the same answer, so prompt variants can be compared fairly.
- **No thinking mode** (`enable_thinking=False`), and at most 150 new tokens (`MAX_NEW_TOKENS`).
- **Refusal clean-up.** Sometimes the model gives an answer and then adds the refusal sentence, which contradicts it. In that case the sentence is removed.

### Model loading

- **Loaded once per command.** `answer_dataset` loads the model once, not once per question.
- **Cache first.** The model is loaded with `local_files_only=True` and only downloaded when it is not cached. This avoids a request to the Hugging Face Hub on every run, which once hung a run for minutes. A failed download ends in a clean error.
- **Model type chosen from the available memory.** When Linux runs out of memory it kills the process (`Killed`, exit code 137), and there is no Python exception to catch. So the available memory is read from `/proc/meminfo` **before** loading:

| Available memory | Type | Why |
|---|---|---|
| ≥ 3.5 GB | float32 | Reads the prompt ~3× faster, because this CPU has no native bfloat16 and emulates it |
| ≥ 2.0 GB | bfloat16 | Half the memory |
| < 2.0 GB | none: clean error, exit 1 | The model would not fit |
| unknown (no `/proc/meminfo`) | bfloat16 | The conservative choice |

`--dtype float32` and `--dtype bfloat16` force a type for testing. If the forced type looks too large for the available memory, a warning is printed, but the model is still loaded: the user decides.

---

## Performance analysis

All recall figures were measured on the public datasets (100 docs questions, 99 code questions) and confirmed with the moulinette. `evaluate` gives the same numbers. With about 100 questions, **one question ≈ 1 %**, so differences of one or two points are noise.

### Current results

Moulinette, chunks of 2,000 characters, k = 10 (`make test_rag_base`):

| Retriever | Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target (@5) |
|---|---|---|---|---|---|---|
| **hybrid** (default) | docs | 0.640 | 0.870 | **0.910** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.545 | 0.778 | **0.828** | 0.859 | ≥ 0.50 ✅ |
| `bm25` | docs | 0.610 | 0.830 | **0.870** | 0.920 | ≥ 0.80 ✅ |
| | code | 0.515 | 0.717 | **0.758** | 0.818 | ≥ 0.50 ✅ |

Times measured with `make test_limits` on a 42 campus computer:

| Step | Time | Subject limit |
|---|---|---|
| `index`: read 1,969 files, chunk them, build and save both indexes and the model | **13.2 s**, including ~4 s to embed the 59,828 windows | 5 min |
| `search_dataset` × 2 (199 questions, each process loading both indexes and the model) | **5.4 s** | 90 s for 200 questions |
| One question, after loading | ~7 ms (hybrid) | |
| Load the BM25 index / the vector index and model | ~0.8 s / ~0.3 s | |
| `answer`: load Qwen3-0.6B | ~5.6 s | |
| `answer`: one question, model load included (float32, ~2,000-token prompt) | ~15–25 s | none |
| `answer_dataset`: 100 questions, model loaded once | ~17 s per question, ~28 min in total | none |

### Evolution of recall@5

| Change | docs | code |
|---|---|---|
| Baseline: fixed-size chunks, plain tokenizer, `K1 = 1.5`, `B = 0.75` | 0.850 | 0.596 |
| `PyChunker`, file path and snake_case terms, `K1 = 1.9`, `B = 0.5` | 0.830 | 0.727 |
| `MdChunker` with the code-fence separator | 0.860 | 0.727 |
| Dotted-name pass removed, `.txt` with `MdChunker`, decorators kept with their `def` | 0.860 | 0.717 |
| Query stopwords (**final BM25**) | 0.870 | 0.758 |
| Hybrid: BM25 top 30 re-ranked with `potion-base-8M` windows (α = 0.5) | **0.910** | **0.828** |

Each change was measured on its own against the previous configuration, and kept only when it helped. The main experiments follow.

### Chunk size

*Baseline configuration.*

| Chunk size | docs @5 | code @5 |
|---|---|---|
| **2000** | **0.850** | 0.596 |
| 1000 | 0.830 | **0.606** |
| 500 | 0.800 | 0.505 |

- Small chunks are more precise, but they split answers.
- Big chunks contain the answer more often, but can fall below IoU 0.05 when the answer is short: a 2,000-character chunk that contains a 40-character answer has an IoU of 0.02.

2,000 was kept: it is the best on docs, and it gives the generator the most context per chunk.

### Ablation of the indexed terms

*Chunks of 2,000, k = 10, recall@5. Each row changes one thing.*

| Configuration | docs @5 | code @5 | Effect of that change |
|---|---|---|---|
| With path, snake_case and dotted passes | 0.830 | 0.727 | |
| − file path | 0.870 | 0.636 | path: **code +9**, docs −4 |
| − snake_case pass | 0.820 | 0.657 | snake_case: **code +7** |
| − dotted pass | 0.830 | 0.717 | neutral → removed |
| `K1/B = 1.5/0.75` | 0.820 | 0.727 | neutral |

The path hurts docs because code chunks take top-5 slots from the documentation: many code paths contain words from docs questions (`vllm/lora/…`, `…/quantization/…`). The code-fence Markdown chunker recovered that loss.

### Markdown chunking variants

*Measured with only the file name in the indexed terms.*

| `.md` chunker | docs @5 |
|---|---|
| Plain fixed-size cuts | 0.860 |
| Heading separators (`#` > `##` > `###`) | 0.820 |
| One chunk per section | 0.820 |
| Headings + code-fence separator | 0.830 |
| **Code-fence separator only (implemented)** | **0.880** |

The questions lost with headings were mostly "right file, IoU < 0.05" cases: short sources whose chunk moved, sometimes to just below the threshold (IoU 0.046, 0.048).

### Query stopwords

| Query filter | docs @5 | code @5 |
|---|---|---|
| None | 0.86 | 0.717 |
| Question words only (17) | 0.86 | 0.747 |
| **Question + common words (~45, implemented)** | **0.87** | **0.758** |
| NLTK English list (~150) | 0.86 | 0.758 |

A longer list does not help. A different list per dataset was deliberately not tried, because the search does not know the type of the question.

### Discarded: chunk overlap

| Overlap | docs @5 | code @5 |
|---|---|---|
| 0 (implemented) | **0.86** | 0.717 |
| 200, all files | 0.83 | 0.737 |
| 400, all files | 0.80 | 0.747 |
| 200, `.py` only | 0.85 | 0.758 |
| 600, `.py` only | 0.85 | 0.727 |

On all files, two almost identical neighbouring chunks take two of the five slots. On `.py` files only, the gain does not grow with the overlap, so it is close to noise.

### Semantic embeddings: choosing the model

Five lightweight CPU models were measured on the real corpus: the time to embed it, and the recall with vectors alone (whole chunks).

| Model | Kind | Size | Whole corpus | docs @5 | code @5 |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2`, 256 tokens | transformer | 88 MB | ~400 s (~20 min on campus) | 0.510 | 0.374 |
| `all-MiniLM-L6-v2`, 128 tokens | transformer | 88 MB | 344 s | 0.460 | 0.333 |
| **`minishlab/potion-base-8M`** | static | **30 MB** | **~2 s** | 0.440 | 0.343 |
| `static-retrieval-mrl-en-v1` | static | 121 MB | 5.2 s | 0.350 | 0.293 |
| `minishlab/potion-retrieval-32M` | static | 125 MB | 2.1 s | 0.340 | 0.263 |

- MiniLM does not fit in the 5-minute indexing limit on a campus computer.
- `potion-base-8M` retrieves about as well as MiniLM cut to 128 tokens, and embeds the corpus ~150× faster, so the vector index can be built inside `index`.
- With whole chunks, **no model beat BM25, alone or fused with it** by reciprocal rank fusion. The tuning below fixed that.

### Semantic embeddings: tuning `potion-base-8M`

A static model represents a text by the **average** of its token vectors, so over a 2,000-character chunk the few words that answer a question are diluted.

| Vectors only | docs @5 | code @5 |
|---|---|---|
| Whole chunks | 0.440 | 0.343 |
| + windows of ~400 characters, best window per chunk | 0.700 | 0.323 |
| + file path in front of every window | **0.730** | **0.596** |
| … with the query stopwords removed | 0.720 | 0.576 |

Windows help most on the docs and the path helps most on the code. Removing the stopwords hurts the vectors, the opposite of BM25: for an embedding, question words carry part of the meaning.

**Combining with BM25.** Reciprocal rank fusion of both lists never improved the code questions: its best setting (vector weight 0.1) gave 0.900 / 0.758, and higher weights lowered both. **Re-ranking the BM25 candidates** improved both. Sweep of the number of candidates N and of α, recall@5 as `docs / code` (BM25 alone: 0.870 / 0.758):

| N \ α | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|
| 10 | 0.920 / 0.788 | 0.920 / 0.798 | 0.920 / 0.798 | 0.920 / 0.778 | 0.910 / 0.788 | 0.910 / 0.778 |
| 20 | 0.910 / 0.788 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.798 | 0.910 / 0.808 | 0.910 / 0.778 |
| **30** | 0.910 / 0.798 | 0.910 / 0.818 | **0.910 / 0.828** | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |
| 50 | 0.910 / 0.798 | 0.910 / 0.818 | 0.910 / 0.828 | 0.910 / 0.808 | 0.910 / 0.818 | 0.910 / 0.778 |

- **+4 points on docs and +7 on code** at recall@5, for ~4 s more indexing and a few milliseconds per question.
- **Not a lucky cell.** Every configuration with 20–50 candidates and α between 0.3 and 0.8 beats BM25. The chosen one is the centre of that plateau, not its best cell.
- **Caveat.** The parameters were tuned on the same public datasets they are measured on. The plateau makes overfitting unlikely, but the private datasets may show a smaller gain. BM25 alone already meets both targets.

### Caching (bonus 4)

**Cold start.** `sentence-transformers` imports PyTorch, and that took ~6.5 s of a ~7.3 s single search. Because the model is just a lookup table and a mean, `StaticEncoder` runs it with numpy, and `index` saves it next to the vectors:

| | With `sentence-transformers` | With `StaticEncoder` |
|---|---|---|
| Load the embedding model | ~6.5 s | **~0.3 s** |
| `index` | 28.2 s | **13.2 s** |
| 199 questions (two processes) | 19.4 s | **5.4 s** |
| Recall@1/3/5/10 | | identical |

Removing the Python loop that averages the token vectors was measured and discarded: `np.add.reduceat` took 13–16 s against 1.6 s for the loop. The work is in the additions, not in the Python iterations.

**Repeated queries.** Measured on the development PC (faster than a campus computer), with the real index:

| | No cache (first run) | Cached |
|---|---|---|
| `search_dataset`, 100 docs questions, `hybrid` | 1.94 s | **0.32 s** |
| `search_dataset`, 100 docs questions, `bm25` | 0.94 s | **0.33 s** |
| `make test_retrieval`: 199 questions, two processes | 4.4 s | **0.7 s** |
| `search`, one question | 0.78 s | **0.33 s** |
| Output and recall@1/3/5/10 | | identical (`diff` of the JSON files) |

What is left in a cached run is starting Python and reading the cache file (~3.7 MB for the 100 docs questions). `make test_limits` still measures the cold case, because `index` removes the caches first. `make test_limits_cached` shows both cases one after the other.

### Generation

Answers cannot be scored automatically, so they were judged by hand. Score: ✅ 2, 🟡 1, ⚪ refusal 0, ❌ −1.

**Battery of 10 questions** (`make test_answer`): 4 docs questions, 4 code questions, 1 off-topic question and 1 trap about a flag that does not exist. Every prompt change was run on the whole battery.

| Prompt version | Score | Confident wrong answers |
|---|---|---|
| Short prompt: "quote the exact names and values" | 7 | 4 |
| + off-topic rule | — | refused valid questions → dropped |
| + few-shot examples | 11 | 1 |
| + new reminder, budget 2,500 | 16 | 0 |
| **+ budget 2,200 (final)** | **17** | **0** |

- **Context budget.** Budgets of 1,500, 2,000, 2,200, 2,500, 2,800 and 3,500 tokens were tried. The answers only change when the *number of chunks* changes. More chunks means more distractors: with 3,500 the model copied values from the wrong chunk.
- **Sampling** refused more valid questions (4 → 6). **Thinking mode** was 3–7× slower (65–150 s per answer) and turned an honest refusal into a confident wrong answer. Both were discarded.

**Full docs dataset.** All 100 questions of `dataset_docs_public.json`, run with `answer_dataset` (BM25 retrieval; the hybrid retriever was added later). Each answer was compared with the reference answer:

| Result | Count | Correct source retrieved (92) | Correct source not retrieved (8) |
|---|---|---|---|
| ✅ correct | **54** | 54 | 0 |
| 🟡 partial or vague | 22 | 21 | 1 |
| ⚪ refusal | 11 | 7 | 4 (honest: the data was missing) |
| ❌ wrong | 13 | 10 | 3 |

- **76 %** of the answers are correct or partial, and **13 %** are wrong (score 117 / 200). An earlier run of the same prompt, with one `answer` call per question, scored 121; the gap is within the uncertainty of judging by hand.
- A correct source was retrieved for 92 of the 100 questions, so most of the loss is in generation. The main failures are over-refusal on "where can I find…" questions, values copied from a nearby but wrong chunk, and answers off the point. These are the "known reasoning limits" of a 0.6B model that the subject mentions.
- When the correct source was not retrieved, the model did not make up an answer in 5 of 8 cases.

### Tests

`make test` runs **1,150 pytest tests** in ~30 s. It never loads Qwen or the embedding model: `tests/conftest.py` replaces the embedding model with a fake bag-of-words encoder in every test.

- **Chunkers:** exact offsets, no gaps or overlaps, never above the maximum, always progressing. These are checked on random texts and on the whole real corpus. Separator priorities, whole code blocks and decorators are tested too.
- **BM25:** scores compared with a naive reference implementation of the formula, plus tie-breaking, query stopwords and the save/load round trip.
- **Vector index and hybrid retriever:** windows, path prefix, best window per chunk, save/load and broken files. The combination formula is checked on fixed scores: `ALPHA = 0` gives exactly the BM25 order, `ALPHA = 1` the similarity order, and only BM25 candidates come out.
- **`CachedRetriever`:** with a stand-in retriever that counts how often it is built and searched. Covers hits and misses, the wrapped retriever never built when everything is cached, another `k` as another entry, copies returned, the round trip to disk, another namespace, broken, empty or non-UTF-8 files, an unwritable folder, a failed final move that keeps the old file, no rewrite when nothing is new, and the namespace changing with the retriever, its parameters and the index files. Through the CLI: a repeated `search` loads no index, `--cache=False` loads every time, and `index` removes the caches.
- **`StaticEncoder`:** a tiny real model built inside the test (5-word tokenizer, table of known vectors). Covers the mean and normalisation computed by hand, unknown words, empty texts (zeros, never NaN), and loading from the cache, with a download, or with neither.
- **Evaluator:** the same numbers as the moulinette.
- **CLI:** every command, argument validation, error messages, unknown or private options, exit codes, `--help`, and source paths outside the project.
- **Mini corpus** in `tests/data`, with easy and hard questions, also run end to end by `make test_rag`.

---

## Design decisions

- **BM25 written from scratch** instead of using a library. The inverted index, idf, saturation and tie-breaking are explicit and tested against the formula.
- **JSON index validated with pydantic**, not pickle. Loading a pickle can run code. The JSON index can be inspected, and a corrupted or foreign file gives a clear error instead of arbitrary behaviour.
- **Two chunkers on one cutting algorithm.** `PyChunker` and `MdChunker` only declare their separators, so the invariants are written and tested once.
- **One change at a time.** Every idea was measured alone: path terms, tokenizer passes, heading cuts, overlap, stopwords, embedding models, fusion methods, context budget and prompt wording. Heading cuts, overlap and list fusion sounded right, and the numbers discarded them.
- **Vectors re-rank BM25 instead of being fused with it.** Fusing the two lists let the vectors bring in chunks with none of the question's words, and lost to BM25 alone. Re-ranking the BM25 top 30 only reorders good candidates.
- **A static embedding model with windows** instead of a transformer like MiniLM. It fits the campus time limit with plenty of room, and windows with max-pooling recover what averaging a whole chunk loses.
- **No deep-learning library for the embeddings.** `potion-base-8M` is a lookup table, so numpy gives the same vectors without PyTorch. The model is saved inside the index folder, so a search depends only on that folder: no network and no Hugging Face cache.
- **One abstract `Retriever`.** `BM25Index`, `HybridRetriever` and `CachedRetriever` share `search(query, k)`. The commands pick one in a single place (`_load_retriever`) and never know which one they got. `VectorIndex` does not depend on BM25: `HybridRetriever` combines them.
- **The query cache is a decorator, not a change to the retrievers.** `CachedRetriever` is a `Retriever` that wraps any other one. It receives a function that builds the wrapped retriever instead of the retriever itself, so nothing is loaded until a question is not cached.
- **A cached result is only reused for the same index and configuration.** Each cache file carries a namespace: the retriever, its parameters (`K1`, `B`, `CANDIDATES`, `ALPHA`, the model) and the size and modification time of the index files. Rebuilding the index or changing a parameter gives a new namespace, and the old results are ignored. Only file metadata is read, so it is instant.
- **The cache never breaks a command.** A missing, broken or foreign cache file means an empty cache, and a cache that cannot be written only prints a warning. It is written to a temporary file and moved over the old one (`os.replace`), so an interrupted write never leaves a half-written file. JSON validated with pydantic, not pickle, as for the index.
- **Exact-match cache, not a semantic one.** Reusing the results of a *similar* question (embedding similarity above a threshold) was considered and discarded. With `potion-base-8M`, *"What is the minimum version of bitsandbytes?"* and *"… maximum version …"* have a similarity of 0.904, the same as a real paraphrase (*"How do I load a LoRA adapter?"* / *"… in vLLM?"*), and *load* / *unload* (0.847) share only 1 of their 5 BM25 results. Because the model averages word vectors, changing one word barely moves the vector, so no threshold separates paraphrases from opposite questions. A semantic cache would also make the output depend on earlier questions, and it would have to load the model on every command to save a ~7 ms search.
- **The lexical method is always available.** `--retriever bm25` runs the mandatory method alone and never loads the vector index.
- **Stopwords on the query only**, so the list can change without rebuilding the index.
- **Paths relative to the project root**, written exactly as in the corpus (`data/raw/vllm-0.10.1/...`), because the grader compares them verbatim.
- **Lazy import and single load of Qwen.** Only `answer` and `answer_dataset` pay for PyTorch, and only once per command.
- **float32 on CPU when it fits.** It reads the prompt ~3× faster than bfloat16. The type is chosen before loading, so a smaller machine gets a slower answer instead of a killed process.
- **A prompt for a small model.** Few positive instructions, examples instead of rules, the question after the context, a fixed refusal sentence that is easy to detect, and greedy decoding so answers are reproducible.
- **No traceback, ever.** Arguments are validated by pydantic before a command runs, and the command line is checked before Fire runs it. Every expected failure is a `RagError` with a message and an exit code: 1 for errors, 2 for usage, 130 for Ctrl+C.
- **Quality gates.** Object-oriented code, type hints everywhere, docstrings on every module, class and function, and `flake8` and `mypy --strict` clean.

---

## Challenges faced

| Challenge | Solution |
|---|---|
| **Fire is permissive.** It ignores type hints, runs a command before rejecting unknown options or a late `--help`, and exposes private methods by name. | `CliCommand.validated` validates every argument with pydantic, and `ArgvGuard` checks the command line against the public commands before Fire runs anything. |
| **Questions and code rarely share words.** | snake_case names are kept whole, the file path is added to the indexed terms, and question words are removed from the query. Code recall@5 went from 0.596 to 0.758 with BM25 alone. |
| **A fix that helps one dataset hurts the other.** The file path gave +9 on code and −4 on docs. | Found with a one-change-at-a-time ablation, and compensated by the code-fence Markdown chunker. |
| **Heading-based Markdown chunking was worse**, even though 79 % of the docs sources start at a heading. | Traced to short sources falling just below IoU 0.05. The code-fence separator gave the best docs recall instead. |
| **Decorators were separated from their function**, because the minimum-fill rule rejected the decorator. | The cut is moved back over the decorator lines, without ever going before the start of the chunk. |
| **Rankings changed between runs.** Equal scores came out in `set` iteration order. | Ranking on `(score, chunk id)`. |
| **The best CPU transformer was too slow**: MiniLM needed ~20 min on campus for the corpus. | A static model, `potion-base-8M`, which embeds the corpus in seconds. |
| **Embeddings lowered recall.** Every model, fused with BM25, scored below BM25 alone. | Short windows with the best window per chunk, the file path in each window, and re-ranking instead of fusion: +4 docs, +7 code. |
| **Cold start.** Importing `sentence-transformers` (PyTorch) took ~6.5 s per command. | The model runs with numpy and is saved with the index. It loads in 0.3 s, with identical vectors and recall. |
| **Slow generation**: 30–40 s per answer. | Profiling showed that bfloat16 is emulated on this CPU, so float32 is used when it fits. |
| **Running out of memory cannot be caught**: the kernel kills the process. | The available memory is checked before loading, which leads to float32, bfloat16 or a clean error. |
| **Evasive answers even with the right context.** | An *oracle test* (only the correct chunk in the context) separated retrieval failures from generation failures. A shorter prompt that asks for exact values fixed them. |
| **"Paris" answered from the model's own knowledge.** An explicit off-topic rule fixed it, but made the model refuse valid questions. | Replaced by a few-shot refused example, so off-topic questions are refused and valid ones are answered. |
| **The prompt leaked into the answers** ("The exact values are: …", the example's `(ServerConfig)`). | A neutral reminder, an example answer with nothing easy to copy at the end, and removal of a refusal sentence that follows a real answer. |
| **A run hung for minutes** on a request to the Hugging Face Hub, even though the model was already cached. | Models are loaded from the cache first, with `local_files_only=True`. |
| **Switching retrievers erased the other one's cache**: with a single cache file, a `bm25` search overwrote the 100 `hybrid` results. | One cache file per retriever (`query_cache_bm25.json`, `query_cache_hybrid.json`). |

---

## Known limitations

- **The re-ranker only reorders.** A chunk that BM25 does not place in its top 30 can never be returned, however close its meaning is.
- **Tuned on the public datasets.** The hybrid parameters come from a stable plateau, but the private datasets may show a smaller gain.
- **A 0.6B model.** It sometimes picks a value from a similar but wrong chunk, and it is sensitive to the exact mix of chunks in its context. The full-dataset answer evaluation was done with BM25 retrieval; with the hybrid retriever, a different set of chunks reaches the model.
- **Grounded is not always correct.** In one case the model quoted vLLM's own assertion message, which contradicts the condition next to it (`vllm/v1/core/kv_cache_coordinator.py`).
- **Memory check in containers.** The check reads the memory of the whole machine (`/proc/meminfo`). Inside a container or cgroup with its own limit, it can overestimate what is available.

---

## Resources

### References

- P. Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401), NeurIPS 2020. The original RAG paper.
- S. Robertson, H. Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf), 2009. The BM25 formula and its parameters.
- C. Manning, P. Raghavan, H. Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/), Cambridge University Press, 2008. Inverted indexes, tf-idf, evaluation with recall.
- G. Cormack, C. Clarke, S. Büttcher, [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009. The fusion method compared against re-ranking.
- [Model2Vec](https://github.com/MinishLab/model2vec) and the [`potion-base-8M` model card](https://huggingface.co/minishlab/potion-base-8M). Static embeddings distilled from a sentence transformer.
- N. Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172), 2023. Why the position of information in the prompt matters.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B). The chat template, the `enable_thinking` switch and the recommended decoding settings.
- Hugging Face Transformers: [chat templates](https://huggingface.co/docs/transformers/chat_templating) and [text generation](https://huggingface.co/docs/transformers/main_classes/text_generation).
- [uv: using PyTorch with uv](https://docs.astral.sh/uv/guides/integration/pytorch/). CPU-only wheels.
- Documentation of [pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [tqdm](https://tqdm.github.io/), [NumPy](https://numpy.org/doc/) and [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers).

### Use of AI

I used an AI assistant throughout the project, as a tutor, a reviewer and a pair programmer:

- **Learning.** Explanations of the subject, RAG, BM25, tokenization, embeddings, PyTorch and transformers, with exercises solved step by step (personal study notes, not part of the repository).
- **Review.** Reviews of my code (bugs found in the reader, the index, the evaluator, the CLI and the answer commands), of my design choices and of the Fire edge cases. It also checked the project against the subject's requirements.
- **Planning of the bonus work.** Step-by-step plans with class skeletons for the vector index, the hybrid retriever and the caching layer, which I then implemented and tested.
- **Code, on request:**
  - the CLI hardening (`CliCommand.validated`);
  - the decorator handling of `PyChunker`;
  - the query stopword list;
  - most of the test suite;
  - several Makefile rules.
- **Experiments.** Scripts for the ablations and measurements reported above (chunk sizes, path terms, Markdown variants, overlap, stopwords, embedding models, fusion and re-ranking sweeps, model types, prompt variants), and the analysis of their results.
- **Documentation.** Drafts of the docstrings and of this README.

I reviewed, ran and understood every part before keeping it, and the design decisions were mine.
