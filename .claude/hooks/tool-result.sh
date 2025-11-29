#!/bin/bash

# HOOK: tool-result.sh
# WHEN: Runs AFTER a tool completes execution
# PURPOSE: React to tool results, run validation, trigger automation
#
# ENVIRONMENT VARIABLES:
#   $TOOL_NAME     - Name of the tool that was used
#   $TOOL_PARAMS   - JSON string of tool parameters
#   $TOOL_RESULT   - Result/output from the tool
#   $PROJECT_ROOT  - Root directory of the project
#
# EXIT CODES:
#   Always exits 0 (cannot block after tool runs)
#
# EXAMPLES:
#   - Run tests after code changes
#   - Update documentation after API changes
#   - Rebuild after dependency changes
#   - Run linters after edits

# Example 1: Auto-run tests after editing Python source files
if [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; then
    FILE_PATH=$(echo "$TOOL_PARAMS" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

    # Check if a Python source file in src/ was modified
    if [[ "$FILE_PATH" == */src/*.py ]]; then
        echo "" >&2
        echo "🧪 Python source changed, running tests..." >&2

        # Run tests (suppress output if successful)
        if command -v poetry &> /dev/null; then
            if poetry run pytest -q 2>&1 | grep -q "passed"; then
                echo "✅ All tests passed" >&2
            else
                echo "⚠️  Some tests failed - review output above" >&2
            fi
        fi
    fi
fi

# Example 2: Run linter after Python file edits
if [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; then
    FILE_PATH=$(echo "$TOOL_PARAMS" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

    if [[ "$FILE_PATH" == *.py ]]; then
        # Quick lint check (only show if errors found)
        if command -v poetry &> /dev/null; then
            LINT_OUTPUT=$(poetry run ruff check "$FILE_PATH" 2>&1)
            if [ ! -z "$LINT_OUTPUT" ]; then
                echo "" >&2
                echo "⚠️  Linting issues found:" >&2
                echo "$LINT_OUTPUT" >&2
            fi
        fi
    fi
fi

# Example 3: Notify when dependencies change
if [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; then
    FILE_PATH=$(echo "$TOOL_PARAMS" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

    if [[ "$FILE_PATH" == *"pyproject.toml" ]]; then
        echo "" >&2
        echo "📦 pyproject.toml changed - you may need to run: poetry install" >&2
    fi
fi

exit 0
