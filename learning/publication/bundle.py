"""Portable figure-reproduction bundles with verified source results."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from learning.publication.analysis import analyze
from learning.publication.figures import render_figures
from learning.publication.io import digest, read_json, write_json


def create_bundle(analysis_dir: Path, destination: Path, *, width: float = 3.5) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    report = read_json(analysis_dir / "analysis.json")
    recipe = read_json(analysis_dir / "recipe.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        runs = []
        for index, source in enumerate(report["sources"]):
            original = Path(source["directory"])
            target = root / "runs" / f"run_{index:04d}"
            target.mkdir(parents=True)
            for name, expected in source["files"].items():
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise ValueError(f"Invalid bundle artifact path: {name}")
                if digest(original / name) != expected:
                    raise ValueError(
                        f"Source changed since analysis: {original / name}"
                    )
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original / name, target / name)
            runs.append(str(target.relative_to(root)))
        write_json(root / "recipe.json", {**recipe, "runs": runs})
        project = Path(__file__).resolve().parents[2]
        code = root / "code"
        (code / "learning").mkdir(parents=True)
        (code / "learning" / "__init__.py").write_text("", encoding="utf-8")
        shutil.copytree(
            project / "learning" / "publication",
            code / "learning" / "publication",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy2(project / name, root / name)
        (root / "reproduce.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "root = Path(__file__).resolve().parent\n"
            'sys.path.insert(0, str(root / "code"))\n'
            "from learning.publication.bundle import reproduce_bundle\n"
            "reproduce_bundle(root)\n",
            encoding="utf-8",
        )
        write_json(root / "bundle_settings.json", {"width": width})
        (root / "README.md").write_text(
            "# Figure reproduction bundle\n\n"
            "Run `uv sync --frozen --extra publication`, then "
            "`uv run --extra publication python reproduce.py`.\n\n"
            "This verifies checksums and regenerates statistics, PDF/SVG plots and "
            "LaTeX tables into `reproduced/`.\n\n"
            "This bundle reproduces analysis from recorded results. It does not "
            "retrain policies or rerun robot trials. Checkpoints and datasets are "
            "identified by hashes in run provenance, not embedded. Source commits, "
            "tracked patches and untracked source snapshots accompany the runs.\n",
            encoding="utf-8",
        )
        write_json(
            root / "checksums.json",
            {
                str(path.relative_to(root)): digest(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
            },
        )
        # Exercise the copied input selection before publishing the bundle directory.
        regenerated = analyze(root / "recipe.json", root / "reproduced")
        for field in ("groups", "paired_differences", "settings"):
            if regenerated[field] != report[field]:
                raise ValueError(
                    "Analysis changed; regenerate the original report before bundling."
                )
        render_figures(
            root / "reproduced" / "analysis.json",
            root / "reproduced" / "figures",
            width=width,
        )
        shutil.rmtree(root / "reproduced")
        shutil.move(str(root), str(destination))


def reproduce_bundle(root: Path) -> None:
    for name, expected in read_json(root / "checksums.json").items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    analyze(root / "recipe.json", root / "reproduced")
    render_figures(
        root / "reproduced" / "analysis.json",
        root / "reproduced" / "figures",
        width=read_json(root / "bundle_settings.json")["width"],
    )
    print(f"Reproduced figures and tables in {root / 'reproduced'}")
