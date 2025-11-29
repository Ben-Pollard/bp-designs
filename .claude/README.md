# Claude Code Configuration Guide

This directory contains project-specific Claude Code configuration.

## Structure

```
.claude/
├── settings.json           # Main configuration file
├── commands/              # Custom slash commands
│   ├── test.md           # /test - Run test suite
│   ├── lint.md           # /lint - Run code quality checks
│   ├── explain.md        # /explain - Detailed explanations
│   ├── refactor.md       # /refactor - Suggest improvements
│   └── setup.md          # /setup - Environment setup
├── hooks/                 # Automation scripts
│   ├── tool-use.sh       # Runs before tool execution
│   ├── tool-result.sh    # Runs after tool completion
│   └── user-prompt-submit.sh  # Runs before message sent
└── README.md             # This file
```

## Settings.json Explained

### Model Selection
Choose which Claude model to use:
- `sonnet` - Best balance of speed and capability (default)
- `opus` - Most capable, slower, more expensive
- `haiku` - Fastest, cheapest, good for simple tasks

### System Prompt
Custom instructions that shape Claude's behavior for this project. This is where you:
- Enforce coding standards
- Set architectural preferences
- Define domain-specific knowledge
- Establish team conventions

### MCP Servers
Extend Claude with external tools:
- **Filesystem**: Controlled file system access
- **Database**: Query databases directly
- **Web**: Fetch web content
- **Custom**: Build your own integrations

Popular MCP servers:
```json
"mcpServers": {
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
  }
}
```

## Slash Commands

Slash commands are markdown files that expand into prompts.

### Creating a Command

1. Create `.claude/commands/mycommand.md`
2. Write the prompt you want Claude to execute
3. Use it with `/mycommand` in chat

### Best Practices
- Keep commands focused on one task
- Include clear instructions
- Specify expected output format
- Add error handling instructions

### Example Command
```markdown
<!-- .claude/commands/deploy.md -->
Deploy the application to production.

1. Run all tests
2. Build the project
3. Run deployment script
4. Verify deployment
5. Report any errors
```

## Hooks

Hooks are bash scripts that run automatically on events.

### Hook Types

#### 1. user-prompt-submit.sh
Runs BEFORE your message is sent to Claude.

**Use cases:**
- Inject context (git status, test results)
- Add environment information
- Validate user input
- Add reminders

**Output:**
- stdout: Appended to your message
- stderr: Shown as notification

#### 2. tool-use.sh
Runs BEFORE Claude executes a tool.

**Use cases:**
- Auto-format before file writes
- Validate operations
- Block dangerous commands
- Enforce project conventions

**Exit codes:**
- 0: Allow tool execution
- 1: Block execution (with error to stderr)

#### 3. tool-result.sh
Runs AFTER a tool completes.

**Use cases:**
- Run tests after code changes
- Lint after edits
- Update docs after API changes
- Notify about dependency changes

**Always exits 0** (cannot block after execution)

### Available Environment Variables

All hooks:
- `$PROJECT_ROOT` - Project directory
- `$TOOL_NAME` - Name of tool (Edit, Bash, Write, etc.)
- `$TOOL_PARAMS` - JSON string of parameters

tool-result.sh only:
- `$TOOL_RESULT` - Output from the tool

user-prompt-submit.sh only:
- `$USER_PROMPT` - Your message text

### Hook Best Practices

1. **Keep them fast** - Hooks run synchronously
2. **Fail gracefully** - Don't crash on missing tools
3. **Clear output** - Use stderr for messages
4. **Test thoroughly** - Bad hooks block your workflow
5. **Document behavior** - Explain what each hook does

### Debugging Hooks

Test hooks manually:
```bash
# Set environment variables
export TOOL_NAME="Write"
export TOOL_PARAMS='{"file_path":"test.py"}'
export PROJECT_ROOT=$(pwd)

# Run hook
bash .claude/hooks/tool-use.sh
echo "Exit code: $?"
```

## Example Workflows

### 1. TDD (Test-Driven Development)
1. Write a test: "Add a test for user authentication"
2. Run `/test` - It fails (expected)
3. Implement feature: "Implement the auth function"
4. Run `/test` - It passes
5. Hook auto-runs tests on file save

### 2. Code Review
1. Make changes to code
2. Run `/lint` - Check quality
3. Run `/refactor` - Get suggestions
4. Run `/test` - Ensure tests pass
5. Commit with confidence

### 3. Learning
1. Find unfamiliar code
2. Use `/explain` - Get detailed breakdown
3. Ask follow-up questions
4. Experiment with changes
5. Tests provide safety net

## Tips for Great AI-Assisted Workflow

1. **Use slash commands** - Faster than typing prompts
2. **Enable hooks** - Automate repetitive tasks
3. **Write tests first** - AI is great at implementation
4. **Ask for explanations** - Understand, don't just accept
5. **Iterate incrementally** - Small changes, frequent validation
6. **Leverage MCP** - Extend capabilities as needed
7. **Customize settings** - Tailor to your team's needs

## Common Issues

### Hooks not running
- Check file permissions: `chmod +x .claude/hooks/*.sh`
- Verify bash path: `which bash`
- Test hooks manually (see Debugging Hooks above)

### Commands not working
- Check filename matches command name
- Verify markdown syntax
- Ensure file is in `.claude/commands/`

### Settings ignored
- Validate JSON syntax
- Check for typos in setting names
- Restart Claude Code after changes

## Learn More

- Claude Code docs: https://github.com/anthropics/claude-code
- MCP documentation: https://modelcontextprotocol.io
- Poetry documentation: https://python-poetry.org
