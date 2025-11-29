#!/bin/bash

# HOOK: tool-use.sh
# WHEN: Runs BEFORE Claude executes any tool
# PURPOSE: Validate, modify, or prevent tool execution
#
# ENVIRONMENT VARIABLES:
#   $TOOL_NAME     - Name of the tool being used (e.g., "Edit", "Bash", "Write")
#   $TOOL_PARAMS   - JSON string of tool parameters
#   $PROJECT_ROOT  - Root directory of the project
#
# EXIT CODES:
#   0 - Allow tool to execute
#   1 - Block tool execution (with error message to stderr)
#
# EXAMPLES:
#   - Auto-format code before Write/Edit
#   - Validate file paths before operations
#   - Prevent destructive operations in production
#   - Add pre-commit style checks

# Example 1: Auto-format Python files before writing
if [ "$TOOL_NAME" = "Write" ] || [ "$TOOL_NAME" = "Edit" ]; then
    # Extract file path from TOOL_PARAMS JSON
    FILE_PATH=$(echo "$TOOL_PARAMS" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

    if [[ "$FILE_PATH" == *.py ]]; then
        echo "🎨 Auto-formatting Python file before write..." >&2

        # Format with black (if available and file exists)
        if command -v poetry &> /dev/null && [ -f "$FILE_PATH" ]; then
            poetry run black "$FILE_PATH" 2>/dev/null || true
        fi
    fi
fi

# Example 2: Warn about destructive operations
if [ "$TOOL_NAME" = "Bash" ]; then
    COMMAND=$(echo "$TOOL_PARAMS" | grep -o '"command":"[^"]*"' | cut -d'"' -f4)

    # Check for potentially destructive commands
    if echo "$COMMAND" | grep -qE "rm -rf|dd |mkfs|format"; then
        echo "⚠️  WARNING: Destructive command detected: $COMMAND" >&2
        # Uncomment to block: exit 1
    fi
fi

# Example 3: Ensure Poetry is used instead of pip
if [ "$TOOL_NAME" = "Bash" ]; then
    COMMAND=$(echo "$TOOL_PARAMS" | grep -o '"command":"[^"]*"' | cut -d'"' -f4)

    if echo "$COMMAND" | grep -qE "^pip install"; then
        echo "❌ ERROR: Use 'poetry add' instead of 'pip install' in this project" >&2
        echo "   Try: poetry add <package-name>" >&2
        exit 1
    fi
fi

# Allow tool to execute
exit 0
