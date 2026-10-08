SHELL := /bin/bash
.SILENT:
ENV_SOURCE=.venv
PYTHON=$(ENV_SOURCE)/bin/python
PIP=$(ENV_SOURCE)/bin/pip

env:
	@python3 -m venv $(ENV_SOURCE)

install: env
	@$(PIP) install -e ".[dev]" > /dev/null

run: install
	@$(PYTHON) src/main.py

debug: install
	@$(PYTHON) -m pdb src/main.py

performance: install
	@$(PYTHON) src/performance.py

destroy:
	rm -rf $(ENV_SOURCE)

re-install: destroy install

clean:
	rm -rf .mypy_cache __pycache__ .pytest_cache build src/fly_in.egg-info

lint: install
	if $(PYTHON) -m flake8 src tests; then \
		echo "Flake8 without issues"; \
	fi; \
	$(PYTHON) -m mypy src tests; \
	true

lint-strict: install
	if $(PYTHON) -m flake8 src tests; then \
		echo "Flake8 without issues"; \
	fi; \
	$(PYTHON) -m mypy --strict src tests; \
	true

.PHONY: env install run debug performance destroy re-install lint lint-strict clean
