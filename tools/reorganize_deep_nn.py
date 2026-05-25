"""Reorganize the Deep-Neural-Network workspace.

Creates `notebooks/` and `projects/` directories, moves Jupyter notebooks
into `notebooks/`, moves Python scripts into `projects/` (skipping this tool
and hidden files), and generates lightweight `README.md` files for each
Python project from the module docstring (if present).

Run this script from the repository root (the folder that contains
`Deep-Neural-Network`) or it will operate on the folder location where
it resides.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_module_docstring(py_path: Path) -> str | None:
    try:
        source = py_path.read_text(encoding="utf8")
        module = ast.parse(source)
        return ast.get_docstring(module)
    except Exception:
        return None


def generate_readme_for_py(py_path: Path, dest_dir: Path) -> None:
    doc = get_module_docstring(py_path)
    title = py_path.stem.replace("_", " ").title()
    readme = dest_dir / py_path.with_suffix("").name / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    contents = f"# {title}\n\n"
    if doc:
        contents += doc.strip() + "\n\n"
    contents += (
        "## Usage\n\nRun the script with Python. Check the module for available\n"
        "functions or add a small CLI wrapper for production use.\n"
    )
    readme.write_text(contents, encoding="utf8")


def reorganize() -> None:
    src = ROOT
    if not src.exists():
        print(f"Directory not found: {src}")
        return

    notebooks_dir = src / "notebooks"
    projects_dir = src / "projects"

    safe_mkdir(notebooks_dir)
    safe_mkdir(projects_dir)

    for entry in src.iterdir():
        if entry.is_dir():
            # keep data, .vscode, and existing projects
            if entry.name in {"data", "projects", "notebooks", ".vscode"}:
                continue
        if entry.suffix == ".ipynb":
            dest = notebooks_dir / entry.name
            print(f"Moving notebook: {entry.name} -> notebooks/")
            shutil.move(str(entry), str(dest))
        elif entry.suffix == ".py":
            # skip this tool file if present
            if entry.name == Path(__file__).name:
                continue
            # skip hidden or config scripts
            if entry.name.startswith("."):
                continue
            print(f"Moving script: {entry.name} -> projects/")
            dest_file = projects_dir / entry.name
            shutil.move(str(entry), str(dest_file))
            # generate README for moved script
            generate_readme_for_py(dest_file, projects_dir)

    print("Reorganization complete. Review changes and commit as desired.")


if __name__ == "__main__":
    reorganize()
