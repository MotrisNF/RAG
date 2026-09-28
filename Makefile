#It's a place-holder.

ARGS ?=

all:
	uv sync

run:
	uv run -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

lint:
	uv run -m flake8 && uv run -m mypy .

lint-strict:
	uv run -m flake8 && uv run -m mypy --strict .

.PHONY: all run debug lint lint-strict