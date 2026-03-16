#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke — Bootstrap
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
###############################################################################
"""
bootstrap.py
============
One-shot setup script: downloads the CorridorKey checkpoint and exports
the TorchScript model ready for Nuke.

Run from the repo root (via install.bat → start.bat, or directly):

    uv run python bootstrap.py

Steps performed
---------------
  1. uv run python nuke/download_checkpoint.py
       Downloads CorridorKey_v1.0.pth (~300 MB) into
       CorridorKey/CorridorKeyModule/checkpoints/

  2. uv run python nuke/export_torchscript.py
       Traces the real GreenFormer model and writes
       Export/CorridorKey.pt

After bootstrap completes, open Nuke and follow the CatFileCreator
instructions printed by export_torchscript.py.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  {' '.join(cmd)}")
    print(sep)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n  ERROR: command exited with code {result.returncode}.")
        sys.exit(result.returncode)


def main() -> None:
    print()
    print("=" * 62)
    print("  CorridorKey — Bootstrap")
    print("=" * 62)

    # Step 1 — download checkpoint
    _run(["uv", "run", "python", "nuke/download_checkpoint.py"])

    # Step 2 — export TorchScript
    _run(["uv", "run", "python", "nuke/export_torchscript.py"])

    print()
    print("=" * 62)
    print("  Bootstrap complete.")
    print("  Export/CorridorKey.pt is ready for Nuke CatFileCreator.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
