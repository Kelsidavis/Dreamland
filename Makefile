# Convenience targets for Dreamland development.
#
# Everything routes through the project's own virtualenv at .venv (created by
# ``pip install -e ".[all,dev]"``) so contributors don't have to think about
# which Python interpreter to call. Override PYTHON= on the make command line
# if you've installed Dreamland system-wide.

PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

.PHONY: help test test-fast lint fmt fix doctor clean

help:
	@echo "Dreamland — common dev targets"
	@echo ""
	@echo "  make test       Run the full pytest suite (~30s on a warm cache)"
	@echo "  make test-fast  Run tests with -x (stop at first failure)"
	@echo "  make lint       ruff check src/ tests/"
	@echo "  make fmt        ruff format src/ tests/"
	@echo "  make fix        ruff check --fix src/ tests/ (auto-fix lints)"
	@echo "  make doctor     Run dreamland doctor against the current env"
	@echo "  make clean      Remove caches and build artefacts"

test:
	$(PYTEST) tests/ -q

test-fast:
	$(PYTEST) tests/ -q -x

lint:
	$(RUFF) check src/dreamland/ tests/

fmt:
	$(RUFF) format src/dreamland/ tests/

fix:
	$(RUFF) check --fix src/dreamland/ tests/

doctor:
	$(PYTHON) -m dreamland.cli.main doctor

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache  -prune -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
