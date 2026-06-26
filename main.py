from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from agent.loop import ResearchAgent
from agent.models import get_configured_model
from agent.openai_client import OpenAIClient
from tui.app import ResearchAgentApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research agent.")
    parser.add_argument("question", nargs="*", help="Question to research.")
    parser.add_argument("--no-tui", action="store_true", help="Run in CLI mode instead of TUI.")
    return parser.parse_args()


async def run_cli(question: str) -> None:
    agent = ResearchAgent(client=OpenAIClient(get_configured_model()))
    state = await agent.run(question)
    print(state.final_answer or "No final answer produced.")
    print()
    print(state.cost_report())


def main() -> None:
    load_dotenv()
    args = parse_args()
    question = " ".join(args.question).strip()

    if not args.no_tui:
        app = ResearchAgentApp(question=question or None)
        app.run()
        return

    if not question:
        raise SystemExit("Provide a question to research.")
    asyncio.run(run_cli(question))


if __name__ == "__main__":
    main()
