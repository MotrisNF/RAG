ARGS ?=

# bash for the `time` keyword used by the rules that run the real program
# (test_rag, test_rag_base, test_answer): it prints how long each command
# took, in the format below.
SHELL := /bin/bash
export TIMEFORMAT := Time: %1Rs

# On 42 computers the home is too small for torch and the model, so the
# virtualenv, the uv cache and the Hugging Face cache go to sgoinfre (shared
# by every computer of the campus). Exported for every rule, so `uv run`
# also finds the venv there. Elsewhere (no sgoinfre) nothing changes.
# Variables already set in the environment win. Other folder: STORE=/path
STORE ?= $(if $(USER),$(wildcard /sgoinfre/students/$(USER)))
ifneq ($(STORE),)
export UV_PROJECT_ENVIRONMENT ?= $(STORE)/rag-venv
export UV_CACHE_DIR           ?= $(STORE)/uv-cache
export HF_HOME                ?= $(STORE)/hf
# Hardlinks do not work on NFS.
export UV_LINK_MODE           ?= copy
endif

all: install

install:
	@uv sync
	@echo "Everything is up to date now."
	@if [ -n "$(STORE)" ]; then echo "venv: $(UV_PROJECT_ENVIRONMENT)"; fi

run:
	@uv run -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

# Local HTTP API (bonus 5) until Ctrl+C; try it at http://127.0.0.1:8000/docs
# Options for `serve`: SERVE_ARGS="--port 8080 --retriever bm25"
SERVE_ARGS ?=

serve:
	@uv run -m src serve $(SERVE_ARGS)

# Each rule keeps its own mypy cache: the cache is only valid for the
# flags that built it, so sharing one made every switch between lint and
# lint-strict re-check torch, transformers... from scratch (~50 s, not ~2 s).
lint:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/lint \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	.

lint-strict:
	uv run -m flake8 . && uv run -m mypy --cache-dir .mypy_cache/strict --strict .

test:
	uv run pytest

# Full pipeline (index -> search_dataset -> evaluate) on the mini corpus in
# tests/data. Generated files go to tests/data/processed and
# tests/data/output (ignored by git, removed by `make clean`).
# Options: TEST_DATASET=dataset_mini_hard.json TEST_K=10 TEST_CHUNK=2000
TEST_DATA    := tests/data
TEST_INDEX   := $(TEST_DATA)/processed
TEST_OUTPUT  := $(TEST_DATA)/output/search_results
TEST_DATASET ?= dataset_mini.json
TEST_K       ?= 5
TEST_CHUNK   ?= 500

test_rag:
	time uv run -m src index --raw_dir $(TEST_DATA)/raw --index_dir $(TEST_INDEX) \
		--max_chunk_size $(TEST_CHUNK)
	time uv run -m src search_dataset \
		--dataset_path $(TEST_DATA)/datasets/UnansweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K) --save_directory $(TEST_OUTPUT) --index_dir $(TEST_INDEX)
	time uv run -m src evaluate \
		--student_search_results_path $(TEST_OUTPUT)/$(TEST_DATASET) \
		--dataset_path $(TEST_DATA)/datasets/AnsweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K)

# Same pipeline on the real exercise data (data/), mimicking the moulinette:
# k=10 and max_chunk_size=2000. The corpus is indexed once and then
# search_dataset + evaluate run twice, for the docs and the code datasets.
BASE_DATA     := data
BASE_INDEX    := $(BASE_DATA)/processed
BASE_OUTPUT   := $(BASE_DATA)/output/search_results/UnansweredQuestions
BASE_DATASETS := dataset_docs_public.json dataset_code_public.json
BASE_K        := 10
BASE_CHUNK    := 2000

test_rag_base:
	time uv run -m src index --raw_dir $(BASE_DATA)/raw --index_dir $(BASE_INDEX) \
		--max_chunk_size $(BASE_CHUNK)
	@for dataset in $(BASE_DATASETS); do \
		echo "===== $$dataset ====="; \
		time uv run -m src search_dataset \
			--dataset_path $(BASE_DATA)/datasets/UnansweredQuestions/$$dataset \
			--k $(BASE_K) --save_directory $(BASE_OUTPUT) \
			--index_dir $(BASE_INDEX) || exit 1; \
		time uv run -m src evaluate \
			--student_search_results_path $(BASE_OUTPUT)/$$dataset \
			--dataset_path $(BASE_DATA)/datasets/AnsweredQuestions/$$dataset \
			--k $(BASE_K) || exit 1; \
	done

# Subject time limits (chapter VII), checked on the real data with the
# moulinette settings. Each rule prints the time and PASS or FAIL.
#   test_index:     indexing the whole corpus in at most 5 minutes.
#   test_retrieval: at most 90 s for 200 questions (both public datasets,
#                   199 questions, each search_dataset loading the index).
#   test_limits:    both.
#   test_limits_cached: test_limits (index removes the query cache, so the
#                   searches are uncached), then test_retrieval again
#                   without indexing (every question read from the cache).
INDEX_LIMIT     := 300
PASS            := printf '\033[1;32mPASS\033[0m\n'
FAIL            := printf '\033[1;31mFAIL\033[0m\n'
RETRIEVAL_LIMIT := 90

test_index:
	@start=$$(date +%s%N); \
	uv run -m src index --raw_dir $(BASE_DATA)/raw --index_dir $(BASE_INDEX) \
		--max_chunk_size $(BASE_CHUNK) || exit 1; \
	ms=$$(( ($$(date +%s%N) - start) / 1000000 )); \
	echo "===== Indexing: $$(LC_ALL=C awk "BEGIN {printf \"%.1f\", $$ms / 1000}") s" \
		"(limit $(INDEX_LIMIT) s)"; \
	if [ $$ms -le $$(( $(INDEX_LIMIT) * 1000 )) ]; then $(PASS); \
	else $(FAIL); exit 1; fi

test_retrieval:
	@questions=0; start=$$(date +%s%N); \
	for dataset in $(BASE_DATASETS); do \
		path=$(BASE_DATA)/datasets/UnansweredQuestions/$$dataset; \
		uv run -m src search_dataset --dataset_path $$path --k $(BASE_K) \
			--save_directory $(BASE_OUTPUT) --index_dir $(BASE_INDEX) \
			|| exit 1; \
		questions=$$(( questions + $$(grep -o '"question_id"' $$path | wc -l) )); \
	done; \
	ms=$$(( ($$(date +%s%N) - start) / 1000000 )); \
	echo "===== Retrieval: $$questions questions in" \
		"$$(LC_ALL=C awk "BEGIN {printf \"%.1f\", $$ms / 1000}") s (limit $(RETRIEVAL_LIMIT) s" \
		"for 200 questions)"; \
	if [ $$ms -le $$(( $(RETRIEVAL_LIMIT) * 1000 )) ]; then $(PASS); \
	else $(FAIL); exit 1; fi

test_limits: test_index test_retrieval

test_limits_cached:
	@echo "===== 1/2: index, then search without cache"
	@$(MAKE) --no-print-directory test_index test_retrieval
	@echo "===== 2/2: search again, no index (query cache)"
	@$(MAKE) --no-print-directory test_retrieval

# Answer battery: runs `answer` on every question of tests/answer_battery
# (one file per question: a QUESTION: line and an EXPECTED: line) and prints
# the expected answer after each one, to compare prompts by eye.
# Needs the index (`make test_rag_base` or `uv run -m src index`).
# Extra options for `answer`: ANSWER_ARGS="--k 5 --dtype bfloat16"
ANSWER_DIR  := tests/answer_battery
ANSWER_ARGS ?=

test_answer:
	@start=$$SECONDS; \
	for file in $(ANSWER_DIR)/*.txt; do \
		question=$$(sed -n 's/^QUESTION: //p' "$$file"); \
		expected=$$(sed -n 's/^EXPECTED: //p' "$$file"); \
		echo "===== $$(basename "$$file" .txt)"; \
		echo "Question: $$question"; \
		time uv run -m src answer "$$question" $(ANSWER_ARGS); \
		echo "Expected: $$expected"; \
		echo; \
	done; \
	echo "Battery total: $$((SECONDS - start)) s"

# Same as test_answer, but on a whole AnsweredQuestions dataset: runs
# `answer` on every question and prints its reference answer after it.
# ~20 s per question (~35 min for 100). Options:
#   ANSWER_DATASET=.../dataset_code_public.json   (default: docs)
#   ANSWER_START=77       first question to run, 1-based (default 1)
#   ANSWER_LIMIT=10       only N questions from there (0 = all)
#   ANSWER_ARGS="..."     extra options for `answer`, as in test_answer
#   ANSWER_LOG=path.log   where the output is also saved (default below,
#                         one file per dataset, overwritten on each run)
ANSWER_DATASET ?= $(BASE_DATA)/datasets/AnsweredQuestions/dataset_docs_public.json
ANSWER_START   ?= 1
ANSWER_LIMIT   ?= 0
ANSWER_LOG     ?= $(BASE_DATA)/output/answer_logs/$(basename $(notdir $(ANSWER_DATASET))).log

test_answer_dataset:
	@mkdir -p $(dir $(ANSWER_LOG)); set -o pipefail; ( \
	mapfile -t lines < <(uv run python -c 'import json, sys; \
		qs = json.load(open(sys.argv[1]))["rag_questions"]; \
		first = int(sys.argv[3]) - 1; n = int(sys.argv[2]); \
		qs = qs[first:first + n] if n else qs[first:]; \
		[print(q["question"].replace("\n", " ") + "\x1f" \
		+ q.get("answer", "").replace("\n", " ")) for q in qs]' \
		$(ANSWER_DATASET) $(ANSWER_LIMIT) $(ANSWER_START)) || exit 1; \
	total=$${#lines[@]}; n=$$(( $(ANSWER_START) - 1 )); start=$$SECONDS; \
	last=$$(( n + total )); \
	for line in "$${lines[@]}"; do \
		n=$$((n + 1)); \
		question=$${line%%$$'\x1f'*}; \
		expected=$${line#*$$'\x1f'}; \
		echo "===== [$$n/$$last]"; \
		echo "Question: $$question"; \
		time uv run -m src answer "$$question" $(ANSWER_ARGS) < /dev/null; \
		echo "Expected: $$expected"; \
		echo; \
	done; \
	echo "Dataset total: $$total questions in $$((SECONDS - start)) s" \
	) 2>&1 | tee $(ANSWER_LOG); \
	echo "Saved to $(ANSWER_LOG)"

clean:
	@find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache $(TEST_INDEX) $(TEST_DATA)/output
	@echo "Cache cleaned"

# Removes the index in data/processed and the json files generated in
# data/output, keeping the folder layout (.gitkeep files).
cleandata:
	@find $(BASE_INDEX) -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
	@find $(BASE_DATA)/output -type f -name '*.json' -delete
	@echo "Index and generated outputs removed"

.PHONY: all install run debug serve lint lint-strict test test_rag test_rag_base test_index test_retrieval test_limits test_limits_cached test_answer test_answer_dataset clean cleandata
