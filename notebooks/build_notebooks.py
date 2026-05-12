"""Convert the jupytext-style .py scripts into proper .ipynb notebooks.

Cell markers used in the source files:
    # %%                  → code cell
    # %% [markdown]       → markdown cell (subsequent comment lines, stripped of '#')

Run:
    python notebooks/build_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Return [(kind, source), ...] in source order."""
    cells: list[tuple[str, list[str]]] = []
    current_kind: str = "code"
    current_body: list[str] = []

    for line in text.splitlines():
        stripped = line.rstrip("\r")
        if stripped.startswith("# %% [markdown]"):
            if current_body:
                cells.append((current_kind, current_body))
            current_kind, current_body = "markdown", []
        elif stripped.startswith("# %%"):
            if current_body:
                cells.append((current_kind, current_body))
            current_kind, current_body = "code", []
        else:
            current_body.append(stripped)
    if current_body:
        cells.append((current_kind, current_body))

    out: list[tuple[str, str]] = []
    for kind, lines in cells:
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()
        if not lines:
            continue
        if kind == "markdown":
            md_lines = []
            for line in lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.startswith("#"):
                    md_lines.append(line[1:])
                else:
                    md_lines.append(line)
            out.append(("markdown", "\n".join(md_lines)))
        else:
            out.append(("code", "\n".join(lines)))
    return out


def py_to_ipynb(py_path: Path, ipynb_path: Path) -> None:
    text = py_path.read_text(encoding="utf-8")
    cells_data = parse_cells(text)
    nb = nbf.v4.new_notebook()
    nb_cells = []
    for kind, src in cells_data:
        if kind == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb_cells.append(nbf.v4.new_code_cell(src))
    nb["cells"] = nb_cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "mimetype": "text/x-python",
        "file_extension": ".py",
    }
    ipynb_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"  {py_path.name} → {ipynb_path.name}  ({len(nb_cells)} cells)")


def main() -> None:
    mapping = {
        "01_cleaning.py": "01_data_cleaning.ipynb",
        "02_eda.py": "02_eda_statistics.ipynb",
        "03_forecasting.py": "03_forecasting.ipynb",
    }
    for src_name, dst_name in mapping.items():
        src = HERE / src_name
        if not src.exists():
            print(f"  skip (missing): {src_name}")
            continue
        py_to_ipynb(src, HERE / dst_name)
    print("Done.")


if __name__ == "__main__":
    main()
