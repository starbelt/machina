# machina — developer entry points. `make help` lists targets.
PY ?= python

.PHONY: help install test lint format check-params determinism clean

help:
	@echo "install       pip install -e .[dev]"
	@echo "test          pytest"
	@echo "lint          ruff check src tests examples"
	@echo "format        ruff format src tests examples"
	@echo "check-params  lint the params CSV fixture (Phase 2)"
	@echo "determinism   two-process build, byte-diff the outputs (Phase 2)"
	@echo "clean         remove build artefacts and caches"

install:
	$(PY) -m pip install -e .[dev]

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests examples

format:
	$(PY) -m ruff format src tests examples

check-params:
	@echo "params pipeline lands in Phase 2 (see vault: Design/Compiler and Params)"; exit 1

determinism:
	@echo "determinism probe lands in Phase 2 (see vault: Design/Roadmap)"; exit 1

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
