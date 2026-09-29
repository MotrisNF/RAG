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

clean:
	@find . -path ./.venv -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache
	@echo "Cache cleaned"

.PHONY: all install run debug lint lint-strict clean