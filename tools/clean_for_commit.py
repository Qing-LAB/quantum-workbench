#!/usr/bin/env python3
"""
clean_for_commit.py — Prepare the repository for a clean commit.

1. Strip outputs and execution counts from all lesson notebooks.
2. Remove generated files in **/figures/ directories.

Usage
-----
    python tools/clean_for_commit.py            # do it
    python tools/clean_for_commit.py --dry-run   # show what would happen
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def strip_notebook(nb_path, dry_run=False):
    """Remove cell outputs and execution counts from a .ipynb file."""
    with open(nb_path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

    if changed:
        size_before = nb_path.stat().st_size
        if dry_run:
            print(f"  would strip: {nb_path}  ({size_before / 1e6:.1f} MB)")
        else:
            with open(nb_path, "w") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
            size_after = nb_path.stat().st_size
            print(f"  stripped: {nb_path}  "
                  f"({size_before / 1e6:.1f} MB -> {size_after / 1e6:.1f} MB)")
    else:
        print(f"  clean:    {nb_path}")


def clean_figures(figures_dir, dry_run=False):
    """Remove all files in a figures/ directory."""
    files = list(figures_dir.iterdir()) if figures_dir.is_dir() else []
    if not files:
        return
    for f in files:
        if dry_run:
            print(f"  would delete: {f}")
        else:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            print(f"  deleted: {f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    # Find all notebooks in lesson directories
    notebooks = sorted(REPO_ROOT.glob("lesson-*/*.ipynb"))
    if notebooks:
        print("Notebooks:")
        for nb in notebooks:
            strip_notebook(nb, dry_run=args.dry_run)
    else:
        print("No lesson notebooks found.")

    # Find all figures directories
    fig_dirs = sorted(REPO_ROOT.glob("lesson-*/figures"))
    if fig_dirs:
        print("\nFigures:")
        for fd in fig_dirs:
            clean_figures(fd, dry_run=args.dry_run)
    else:
        print("\nNo figures directories found.")

    if args.dry_run:
        print("\n(dry run — no changes made)")


if __name__ == "__main__":
    main()
