"""Publication-sized figures and LaTeX tables from audited analysis outputs."""

from __future__ import annotations

import csv
from pathlib import Path

from learning.publication.io import digest, read_json, write_json


def _latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_figures(analysis_path: Path, output: Path, *, width: float = 3.5) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if width <= 0:
        raise ValueError("Figure width must be positive inches.")
    report = read_json(analysis_path)
    output.mkdir(parents=True, exist_ok=True)
    groups = report["groups"]
    rows = []
    for group in groups:
        for run in group["individual_runs"]:
            rows.append(
                {
                    **{k: group[k] for k in ("task", "condition", "split", "method")},
                    **{
                        k: run[k]
                        for k in ("run_id", "training_seed", "score", "episodes")
                    },
                }
            )
    with (output / "source_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    panels = sorted({(g["task"], g["condition"], g["split"]) for g in groups})
    artifacts = []
    with plt.rc_context(settings):
        for index, panel in enumerate(panels):
            selected = [
                g for g in groups if (g["task"], g["condition"], g["split"]) == panel
            ]
            fig, ax = plt.subplots(
                figsize=(width, max(1.8, 0.5 * len(selected) + 0.7)),
                layout="constrained",
            )
            for y, group in enumerate(selected):
                scores = [run["score"] for run in group["individual_runs"]]
                ax.scatter(scores, [y] * len(scores), s=12, color="0.65", zorder=2)
                ax.plot(
                    [group["low"], group["high"]], [y, y], color="#177E89", linewidth=2
                )
                ax.scatter(
                    [group["mean"]], [y], marker="D", color="#A23B45", s=24, zorder=3
                )
            ax.set_yticks(range(len(selected)), [g["method"] for g in selected])
            ax.set_xlabel(report["metric"])
            ax.set_title(" / ".join(panel), fontsize=9)
            ax.grid(axis="x", alpha=0.2)
            ax.invert_yaxis()
            for suffix in ("pdf", "svg", "png"):
                path = output / f"comparison_{index:02d}.{suffix}"
                fig.savefig(path, dpi=200)
                artifacts.append(path.name)
            plt.close(fig)
    table = [
        r"\begin{tabular}{llllrr}",
        r"\hline",
        r"Task & Condition & Split & Method & Mean [CI] & Runs \\",
        r"\hline",
    ]
    for group in groups:
        labels = " & ".join(
            _latex(str(group[k])) for k in ("task", "condition", "split", "method")
        )
        table.append(
            labels
            + f" & {group['mean']:.3f} [{group['low']:.3f}, {group['high']:.3f}]"
            + f" & {group['runs']} "
            + r"\\"
        )
    table += [r"\hline", r"\end{tabular}"]
    (output / "results.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    write_json(
        output / "figure_manifest.json",
        {
            "schema_version": 1,
            "analysis_sha256": digest(analysis_path),
            "analysis": str(analysis_path.resolve()),
            "width_inches": width,
            "style": settings,
            "confidence": report["settings"]["confidence"],
            "run_ids": [s["run_id"] for s in report["sources"]],
            "artifacts": {
                name: digest(output / name)
                for name in [*artifacts, "source_data.csv", "results.tex"]
            },
        },
    )
