# Contributing to TEMPO

Thank you for your interest in contributing to TEMPO!

## Development Setup

1. Fork and clone the repository
2. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting:
  ```bash
  black tempo/
  ```
- Run flake8 for linting:
  ```bash
  flake8 tempo/
  ```

## Testing

Run tests before submitting a pull request:
```bash
pytest
```

## Submitting Changes

1. Create a new branch for your changes
2. Make your changes and commit with descriptive messages
3. Run tests and linters
4. Push to your fork and submit a pull request

## Paper Contributions

For contributions to the academic paper:

1. LaTeX files are in the `paper/` directory
2. Build the paper with `make` in the `paper/` directory
3. Follow academic writing standards
4. Include references in BibTeX format in `references.bib`

## Questions?

Feel free to open an issue for any questions or suggestions.
