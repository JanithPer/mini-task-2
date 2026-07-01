from __future__ import annotations

from pathlib import Path


def write_cumulative_cost_plot(per_iteration_costs: list[float], output_path: Path) -> Path | None:
    if not per_iteration_costs:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cumulative: list[float] = []
    running_total = 0.0
    for cost in per_iteration_costs:
        running_total += cost
        cumulative.append(running_total)

    iterations = list(range(1, len(cumulative) + 1))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, cumulative, marker="o", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative cost (USD)")
    ax.set_title("Agent cumulative cost by iteration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
