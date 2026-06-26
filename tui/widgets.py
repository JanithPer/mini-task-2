from __future__ import annotations

from textual.widgets import TextArea


class TraceLog(TextArea):
    def __init__(self, **kwargs):
        super().__init__(text="", **kwargs)

    def on_mount(self) -> None:
        self.border_title = "Live Trace"
        self.read_only = True
        self.show_line_numbers = False

    def append(self, text: str) -> None:
        if self.text:
            self.text += "\n" + text
        else:
            self.text = text
        lines = len(self.document.lines)
        self.cursor_location = (lines - 1, 0)


class StatusPanel(TextArea):
    def __init__(self, **kwargs):
        super().__init__(text="", **kwargs)

    def on_mount(self) -> None:
        self.border_title = "Status"
        self.read_only = True
        self.show_line_numbers = False

    def update_status(
        self,
        iteration: int = 0,
        max_iterations: int = 30,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        runtime: float = 0.0,
        tool_calls: int = 0,
        phase: str = "idle",
    ) -> None:
        total = input_tokens + output_tokens
        self.text = "\n".join([
            f"Phase:       {phase}",
            "",
            f"Iteration:   {iteration} / {max_iterations}",
            "",
            f"Input:       {input_tokens:,} tok",
            f"Output:      {output_tokens:,} tok",
            f"Total:       {total:,} tok",
            "",
            f"Cost:        ${cost:.4f}",
            f"Runtime:     {runtime:.1f}s",
            f"Tool Calls:  {tool_calls}",
        ])


class AnswerPanel(TextArea):
    def __init__(self, **kwargs):
        super().__init__(text="", **kwargs)

    def on_mount(self) -> None:
        self.border_title = "Final Answer"
        self.read_only = True
        self.show_line_numbers = False
        self.styles.display = "none"

    def show_answer(self, answer: str) -> None:
        self.styles.display = "block"
        self.text = answer
