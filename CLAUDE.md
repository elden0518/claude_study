# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Examples

All examples are standalone scripts run directly with the virtual environment Python:

```bash
# Activate venv first (Windows)
.venv\Scripts\activate

# Then run any example
python 01_basics/01_hello_claude.py
python 02_advanced/07_tool_use.py
python 04_project/cli_assistant.py
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

## Model ID Convention

This project uses a proxy-prefixed model ID: `"ppio/pa/claude-sonnet-4-6"` — not the bare `"claude-sonnet-4-6"`. All new code in this repo must use this format.

## Architecture

The project is a sequential learning curriculum with 5 modules:

| Module | Path | Purpose |
|--------|------|---------|
| Basics | `01_basics/` | Core API: client init, messages, parameters, streaming, error handling |
| Advanced | `02_advanced/` | Tool use, vision, conversation management, structured output |
| MCP & Extensions | `03_mcp_skill_command/` | MCP server development with FastMCP, skills, slash commands |
| Project | `04_project/cli_assistant.py` | Integrates all modules into a full CLI assistant |
| Production | `05_production/` | Extended Thinking, Prompt Caching, Batch API, Async Client, Token Counting |

Each file is self-contained with extensive Chinese comments explaining every step. There are no shared utility modules — code patterns are intentionally repeated across files for learning clarity.

## Windows Encoding Fix

Every script starts with this block to handle Windows GBK console encoding for Chinese/emoji output:

```python
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
```

New scripts should include this at the top.

## Key Dependencies

- `anthropic` — Claude API SDK (`client.messages.create()`, `client.messages.stream()`)
- `fastmcp` — MCP server framework (`FastMCP`, `@mcp.tool()`, `@mcp.resource()`)
- `pydantic` — structured output validation
- `python-dotenv` — `.env` loading via `load_dotenv()`