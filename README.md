# Research Agent

A framework-free Python research agent that plans, searches the web, reads and writes workspace files, executes Python and Bash in an ephemeral Docker sandbox, tracks usage, and streams execution trace events into a Textual TUI.

![TUI Screenshot](Screenshot.png)

## Setup

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

Fill in `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env`.

Build the sandbox image:

```powershell
docker build -t research-agent-sandbox sandbox
```

Run the CLI:

```powershell
uv run python main.py "Analyse the last 5 years of Tesla earnings and predict Q3 revenue"
```

Run the TUI:

```powershell
uv run python main.py
```

Run tests:

```powershell
uv run pytest
```

## Architecture

```
main.py
  ├── ResearchAgentApp (Textual TUI)  or  run_cli()
  │     └── ResearchAgent.run(question)
  │           ├── Planner.create_plan()
  │           └── LOOP (max 30 iterations)
  │                 ├── OpenAIClient.decide()  →  ModelDecision
  │                 ├── CostTracker.update()
  │                 ├── Hooks.before_tool() / before_final()
  │                 ├── ToolRegistry.execute()
  │                 │     ├── WebSearchTool (Tavily)
  │                 │     ├── ReadFileTool / WriteFileTool
  │                 │     ├── PythonExecutorTool → Docker sandbox
  │                 │     └── BashExecutorTool  → Docker sandbox
  │                 └── Events → event_sink (TUI / stdout)
  └── _write_trace_summary() → traces/trace-*.json
```

## Tools

| Tool | Description | Security |
|---|---|---|
| `web_search` | Web search via Tavily API | API-key gated |
| `read_file` | Read files from workspace | Path traversal prevented |
| `write_file` | Write files to workspace | Path traversal prevented |
| `execute_python` | Run Python code in Docker | `--network none`, 30s timeout |
| `execute_bash` | Run shell commands in Docker | `--network none`, 30s timeout |

## Guardrails (Hooks)

- **Iteration limit** — hard stop at 30 iterations
- **Tool validation** — rejects unknown tool names
- **Answer without code** — forces code execution for analytical questions
- **Research completeness** — requires web search + code + report
- **Chart required** — requires a generated chart for analysis tasks
- **Citation validation** — requires source URLs in final answer

## Notes

- Python and Bash tools run only through Docker (image: `research-agent-sandbox`).
- The container is ephemeral, network disabled, memory limited to 512 MB, and CPU limited to 1.
- File tools are restricted to the configured workspace.
- The agent stops after 30 iterations.
- Trace JSON files are written to `traces/` after each run.
- Supports OpenAI-compatible providers (OpenAI, Gemini via OpenAI-compatible endpoint).
