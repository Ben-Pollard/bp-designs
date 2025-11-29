Set up the development environment for this project.

Execute these steps:
1. Check if Poetry is installed (`poetry --version`)
2. Install dependencies: `poetry install`
3. Verify installation: `poetry run python -c "import bp_designs; print('Success!')"`
4. Run initial tests: `poetry run pytest`
5. Check code quality: `poetry run ruff check src`

If any step fails:
- Explain the error
- Provide troubleshooting steps
- Offer to help resolve the issue

After successful setup, explain:
- How to activate the virtual environment (`poetry shell`)
- Available commands (`/test`, `/lint`, etc.)
- Next steps for development
