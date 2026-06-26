from __future__ import annotations

from agent.prompts import PLANNER_PROMPT


class Planner:
    def __init__(self, client: object) -> None:
        self.client = client

    async def create_plan(self, question: str) -> str:
        if not hasattr(self.client, "complete_text"):
            return self.fallback_plan(question)
        return await self.client.complete_text(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": question},
            ]
        )

    @staticmethod
    def fallback_plan(question: str) -> str:
        return "\n".join(
            [
                f"1. Clarify the research target: {question}",
                "2. Search for recent and historical source data.",
                "3. Save useful facts, source URLs, and tabular data.",
                "4. Run Python analysis on the collected data.",
                "5. Generate charts when trends or forecasts are involved.",
                "6. Write a markdown report with citations.",
                "7. Validate completeness before final answer.",
            ]
        )

