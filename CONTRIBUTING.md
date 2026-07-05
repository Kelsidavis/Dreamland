# Contributing to Dreamland

Don't Panic. Contributions are welcome.

## Development Setup

```bash
git clone https://github.com/Kelsidavis/Dreamland.git
cd Dreamland
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # full suite
pytest tests/test_foo.py  # single file
pytest -x                 # stop on first failure
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
```

## Adding a Skill

```bash
dreamland skill-init my_skill    # generates ~/.dreamland/skills/my_skill_skill.py
```

Edit the generated file, restart Dreamland, and your skill is loaded.

## Pull Requests

1. Fork and branch from `main`
2. Write tests for new features
3. Run `pytest` and `ruff check` before pushing
4. Open a PR with a clear description
