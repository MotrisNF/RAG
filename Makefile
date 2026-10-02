ARGS ?=

all: install

install:
	@uv sync
	@echo "Everything is up to date now."

run:
	@uv run -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run -m flake8 && uv run -m mypy \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	.

lint-strict:
	uv run -m flake8 && uv run -m mypy --strict .

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
	uv run -m src index --raw_dir $(TEST_DATA)/raw --index_dir $(TEST_INDEX) \
		--max_chunk_size $(TEST_CHUNK)
	uv run -m src search_dataset \
		--dataset_path $(TEST_DATA)/datasets/UnansweredQuestions/$(TEST_DATASET) \
		--k $(TEST_K) --save_directory $(TEST_OUTPUT) --index_dir $(TEST_INDEX)
	uv run -m src evaluate \
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
	uv run -m src index --raw_dir $(BASE_DATA)/raw --index_dir $(BASE_INDEX) \
		--max_chunk_size $(BASE_CHUNK)
	@for dataset in $(BASE_DATASETS); do \
		echo "===== $$dataset ====="; \
		uv run -m src search_dataset \
			--dataset_path $(BASE_DATA)/datasets/UnansweredQuestions/$$dataset \
			--k $(BASE_K) --save_directory $(BASE_OUTPUT) \
			--index_dir $(BASE_INDEX) || exit 1; \
		uv run -m src evaluate \
			--student_search_results_path $(BASE_OUTPUT)/$$dataset \
			--dataset_path $(BASE_DATA)/datasets/AnsweredQuestions/$$dataset \
			--k $(BASE_K) || exit 1; \
	done

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

.PHONY: all install run debug lint lint-strict test test_rag test_rag_base clean cleandata
