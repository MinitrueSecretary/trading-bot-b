# CLAUDE.md - Python Project Guidelines

## Build & Environment Commands
- Setup Environment: `uv venv`
- Activate Environment: `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
- Install Dependencies: `uv pip install -e .` or `uv sync`
- Add Dependency: `uv add <package>`
- Run Application: `uv run python main.py`

## Code Quality & Linting Commands
- Lint & Format (Ruff): `ruff check . --fix` and `ruff format .`
- Type Checking (Mypy): `mypy . --strict`

## Testing Commands
- Run All Tests: `pytest`
- Run with Coverage: `pytest --cov=src`
- Run Specific Test File: `pytest tests/test_filename.py`

## Coding Conventions
- Style Guide: Adhere strictly to PEP 8 standards.
- Type Hints: Mandatory PEP 484 type hints for all function signatures. Avoid using `Any`.
- Docstrings: Use Google-style docstrings for all modules, classes, and public functions.
- Strings: Use f-strings for formatting, use single quotations for single-line strings.


## DO NOT (Anti-Patterns to Avoid)
- NEVER use wildcard imports (`from module import *`).
- NEVER use mutable objects (like lists or dicts) as default arguments in functions.
- NEVER commit hardcoded secrets, API keys, or `.env` files to git.
- NEVER bypass linting or test steps before finishing a task.