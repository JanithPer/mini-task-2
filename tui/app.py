from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input

from agent.loop import ResearchAgent
from agent.models import get_configured_model
from agent.openai_client import OpenAIClient
from tui.events import AnswerEvent, CostEvent, ThoughtEvent, ToolCallEvent, ToolResultEvent, TraceEvent
from tui.widgets import AnswerPanel, StatusPanel, TraceLog


class ResearchAgentApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #input_row {
        height: 3;
        padding: 0 1;
        dock: top;
    }
    Input {
        width: 1fr;
    }
    Button {
        width: 10;
        margin-left: 1;
    }
    #body {
        height: 1fr;
    }
    #trace_log {
        width: 2fr;
        border: solid $accent;
        padding: 0 1;
    }
    #side_panel {
        width: 1fr;
    }
    #status_panel {
        height: auto;
        border: solid $secondary;
        padding: 1;
    }
    #answer_panel {
        height: 1fr;
        border: solid $success;
        padding: 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+s", "save_trace", "Save Trace"),
    ]

    def __init__(self, question: str | None = None) -> None:
        super().__init__()
        self._question = question
        self.trace_log: TraceLog | None = None
        self.status_panel: StatusPanel | None = None
        self.answer_panel: AnswerPanel | None = None
        self._start_time: float = 0.0
        self._tool_call_count: int = 0
        self._trace_lines: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="input_row"):
            yield Input(placeholder="Enter a research question...", id="prompt")
            yield Button("Run", variant="primary", id="run")
        with Horizontal(id="body"):
            yield TraceLog(id="trace_log")
            with Vertical(id="side_panel"):
                yield StatusPanel(id="status_panel")
                yield AnswerPanel(id="answer_panel")
        yield Footer()

    def on_mount(self) -> None:
        load_dotenv()
        self.trace_log = self.query_one("#trace_log", TraceLog)
        self.status_panel = self.query_one("#status_panel", StatusPanel)
        self.answer_panel = self.query_one("#answer_panel", AnswerPanel)
        if self._question:
            self.query_one("#prompt", Input).value = self._question
            self.run_agent(self._question)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            return
        prompt = self.query_one("#prompt", Input).value.strip()
        if not prompt:
            return
        self.run_agent(prompt)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "prompt" and event.value.strip():
            self.run_agent(event.value.strip())

    def action_save_trace(self) -> None:
        path = Path(f"trace-{int(time.time())}.txt")
        path.write_text("\n".join(self._trace_lines), encoding="utf-8")
        self._trace(f"Trace saved to: {path.resolve()}")

    def run_agent(self, question: str) -> None:
        self._start_time = time.monotonic()
        self._tool_call_count = 0
        self._trace_lines = []
        self.answer_panel.styles.display = "none"
        self.answer_panel.text = ""
        self.trace_log.text = ""
        self.status_panel.update_status(phase="starting...")
        asyncio.create_task(self._run_agent(question))

    async def _run_agent(self, question: str) -> None:
        self._trace(f"Question: {question}")

        client = OpenAIClient(get_configured_model())
        agent = ResearchAgent(client=client, event_sink=self._handle_event)

        self._start_time = time.monotonic()

        try:
            state = await agent.run(question)
        except Exception as exc:
            self._trace(f"Error: {exc}")
            self.status_panel.update_status(phase="error")
            return

        self._trace("")
        self._trace("Run complete.")
        self._trace("")
        self._trace(state.cost_report())
        self.status_panel.update_status(
            iteration=state.iteration_count,
            input_tokens=state.token_usage.input_tokens,
            output_tokens=state.token_usage.output_tokens,
            cost=state.estimated_cost,
            runtime=state.runtime_seconds(),
            tool_calls=len(state.tool_calls),
            phase="done",
        )

    def _plain(self, markup: str) -> str:
        return Text.from_markup(markup).plain

    def _trace(self, text: str) -> None:
        plain = self._plain(text)
        self.trace_log.append(plain)
        self._trace_lines.append(plain)

    def _handle_event(self, event: TraceEvent) -> None:
        self._render_event(event)

    def _render_event(self, event: TraceEvent) -> None:
        if isinstance(event, ThoughtEvent):
            if event.thought.strip():
                self._trace(f"\nThought: {event.thought}")
                self._trace("")

        elif isinstance(event, ToolCallEvent):
            self._tool_call_count += 1
            payload = json.dumps(event.payload, indent=2) if event.payload else "{}"
            self._trace(f"Tool: {event.name}")
            self._trace(f"{payload}")
            self._trace("")

        elif isinstance(event, ToolResultEvent):
            status = "Success" if event.ok else "Error"
            output = str(event.output)
            if len(output) > 500:
                output = output[:500] + "..."
            self._trace(f"{status}: {event.name}")
            self._trace(f"{output}")
            self._trace("")

        elif isinstance(event, AnswerEvent):
            self._trace(f"\nAnswer: {event.answer}")
            self.answer_panel.show_answer(event.answer)

        elif isinstance(event, CostEvent):
            runtime = time.monotonic() - self._start_time
            self.status_panel.update_status(
                iteration=event.iteration,
                input_tokens=event.input_tokens,
                output_tokens=event.output_tokens,
                cost=event.cost,
                runtime=runtime,
                tool_calls=self._tool_call_count,
                phase="running",
            )
