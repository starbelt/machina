# machina — developer entry points. `make help` lists targets.
PY ?= python

.PHONY: help install test lint format check-params determinism clean

help:
	@echo "install       pip install -e .[dev]"
	@echo "test          pytest"
	@echo "lint          ruff check src tests examples"
	@echo "format        ruff format src tests examples"
	@echo "check-params  machina params check over the fixture table"
	@echo "determinism   build the probe artifacts in two processes, byte-diff them"
	@echo "clean         remove build artefacts and caches"

install:
	$(PY) -m pip install -e .[dev]

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests examples scripts

format:
	$(PY) -m ruff format src tests examples scripts

check-params:
	cd tests/params/fixtures && $(PY) -m machina params check --csv params.csv --modules fixture_params

# Two separate PROCESSES with different hash seeds: one process would share a graph built once
# and test only the writer. The failure this catches is an ordering that came from a set.
determinism:
	rm -rf build/det-a build/det-b
	PYTHONHASHSEED=1 $(PY) scripts/determinism_probe.py --out build/det-a
	PYTHONHASHSEED=2 $(PY) scripts/determinism_probe.py --out build/det-b
	diff -r build/det-a build/det-b && echo "artifacts are byte-identical across two processes"

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
