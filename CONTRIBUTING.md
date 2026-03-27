# Contributing

## Development Setup
1. Create a virtual environment and activate it.
2. Install the project in editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
```

## Common Commands
```bash
pytest
ruff check .
python -m phops validate-config -c config.yaml
python -m phops run -c config.yaml
```

## Guidelines
- Keep the service layer reusable from both CLI and GUI code.
- Do not mix plotting, subprocess execution, and config parsing into scientific logic unless there is a clear adapter boundary.
- Add tests for new behavior, especially around config parsing and pipeline control flow.
- Prefer small, reviewable commits with clear changelog entries.
