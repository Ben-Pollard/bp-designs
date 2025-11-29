Run all code quality checks in sequence.

Execute these commands:
1. `poetry run ruff check src tests` - Fast linting for errors and style
2. `poetry run black --check src tests` - Check code formatting
3. `poetry run mypy src` - Type checking

For each tool:
- Show the output
- Explain any errors or warnings found
- Offer to fix issues automatically if appropriate

If all checks pass, confirm the code meets quality standards.
If any fail, provide a summary and ask if I should fix the issues.
