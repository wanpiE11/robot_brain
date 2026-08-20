# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python demo for a plan-and-execute robot agent. Key paths:

- `main.py`: entry point that builds the agent and runs the demo task.
- `skills.py`: mock robot skills and shared `RobotState`.
- `model_trace.py`: local tracing and token/debug logging for LLM calls.
- `rai/`: vendored robot-agent library code; treat as internal dependency code.
- `learning_notes/`: design docs and target architecture notes.
- `config.toml` and `.env.example`: model, tracing, and environment setup.

## Build, Test, and Development Commands
- `uv sync`: install Python dependencies and create/update the local virtual environment.
- `uv run python main.py`: run the demo agent end to end.
- `cp .env.example .env`: create local environment settings before running.

This project does not currently ship a formal automated test suite or lint command.

## Coding Style & Naming Conventions
Use standard Python 3.10+ style with 4-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes and Pydantic models. Keep edits aligned with existing patterns in `rai/` and the top-level scripts. Prefer small, explicit functions over new abstractions unless they reduce repeated agent logic. If you add tooling, document it in `pyproject.toml` and keep configuration centralized.

## Testing Guidelines
No pytest configuration is present yet. When you add behavior, include a minimal reproducible check alongside the change, such as a focused script path, a logged trace, or a small unit test if the repo gains one. For agent-flow changes, verify the demo still runs with `uv run python main.py` and that the plan/replan loop terminates as expected.

## Commit & Pull Request Guidelines
Git history uses short, imperative messages with optional prefixes such as `fix:` and `docs:`. Follow that style, for example: `fix: avoid replanner loop`. Pull requests should summarize the behavior change, note any model or config updates, and include output samples or trace notes when execution behavior changes.

## Security & Configuration Tips
Do not commit real API keys. Copy `.env.example` to `.env` locally and keep provider credentials out of version control. Review `config.toml` before changing model providers or tracing settings.
