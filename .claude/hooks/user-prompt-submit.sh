#!/bin/bash

# HOOK: user-prompt-submit.sh
# WHEN: Runs BEFORE your message is sent to Claude
# PURPOSE: Inject context, validate input, add automation
#
# ENVIRONMENT VARIABLES:
#   $USER_PROMPT   - The message you typed
#   $PROJECT_ROOT  - Root directory of the project
#
# OUTPUT:
#   Anything written to stdout is APPENDED to your message
#   Anything written to stderr is shown as a notification
#
# EXIT CODES:
#   0 - Continue with message
#   1 - Block message from being sent
#
# EXAMPLES:
#   - Add git status to every message
#   - Include recent error logs
#   - Inject environment info
#   - Add TODO reminders

# Example 1: Add git status if this becomes a git repo
if [ -d .git ]; then
    GIT_STATUS=$(git status --short 2>/dev/null)
    if [ ! -z "$GIT_STATUS" ]; then
        echo "" >&2
        echo "📊 Git status:" >&2
        echo "$GIT_STATUS" >&2
    fi
fi

# Example 2: Remind about uncommitted changes
if [ -d .git ]; then
    UNCOMMITTED=$(git diff --stat 2>/dev/null | tail -1)
    if [ ! -z "$UNCOMMITTED" ]; then
        echo "💡 You have uncommitted changes" >&2
    fi
fi

# Example 3: Check if dependencies are installed
if [ -f pyproject.toml ] && [ ! -d .venv ] && [ ! -d $(poetry env info -p 2>/dev/null) ]; then
    echo "⚠️  Poetry environment not initialized. Run: poetry install" >&2
fi

# Example 4: Inject recent test results into context (disabled by default)
# Uncomment to automatically include test results in every message
# if command -v poetry &> /dev/null; then
#     echo ""
#     echo "<recent_test_results>"
#     poetry run pytest --tb=no -q 2>&1 | tail -5
#     echo "</recent_test_results>"
# fi

exit 0
