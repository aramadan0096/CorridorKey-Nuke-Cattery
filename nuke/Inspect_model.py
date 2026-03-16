#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
###############################################################################
"""
inspect_model.py
================
Diagnostic tool.  Run this whenever the export fails with a class-selection
or load_state_dict error.  It prints:

  - Every nn.Module class found in model_transformer.py
  - Each class's immediate child sub-module names
  - The top-level sub-module names present in the checkpoint
  - Which class the discovery algorithm would select

Usage:
    uv run python nuke/inspect_model.py
    uv run python nuke/inspect_model.py --checkpoint path/to/CorridorKey.pth
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT  = Path(__file__).resolve().parent.parent
_INNER_ROOT = _REPO_ROOT / "CorridorKey"   # Python package root (CorridorKeyModule lives here)
sys.path.insert(0, str(_INNER_ROOT))       # for CorridorKeyModule imports
sys.path.insert(0, str(_REPO_ROOT))        # for nuke.nuke_wrapper imports

from nuke.nuke_wrapper import _strip_orig_mod, _discover_model_class  # type: ignore


def _top_level_names(net: nn.Module) -> list[str]:
    return sorted(name for name, _ in net.named_children())


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect model_transformer.py classes")
    p.add_argument(
        "--checkpoint",
        default=str(_INNER_ROOT / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"),
    )
    args = p.parse_args()

    print()
    print("=" * 64)
    print("  CorridorKey — model_transformer.py inspection")
    print("=" * 64)

    # ── Import module ─────────────────────────────────────────────────
    try:
        mod = importlib.import_module("CorridorKeyModule.core.model_transformer")
    except ImportError as e:
        print(f"\n  ERROR: could not import CorridorKeyModule.core.model_transformer\n  {e}")
        print("  Run: uv sync")
        sys.exit(1)

    print(f"  Module file: {mod.__file__}")

    # ── List all public nn.Module subclasses ──────────────────────────
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if name.startswith("_"):
            continue
        if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
            continue
        if getattr(obj, "__module__", "") != mod.__name__:
            continue
        candidates.append((name, obj))

    print(f"\n  Found {len(candidates)} public nn.Module subclass(es):\n")

    for cls_name, cls in candidates:
        print(f"  ┌─ {cls_name}")
        try:
            sig = inspect.signature(cls.__init__)
            req = [
                p.name for p in list(sig.parameters.values())[1:]
                if p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if req:
                print(f"  │  Required constructor args: {req}  (cannot auto-instantiate)")
                print(f"  └─")
                continue
            instance = cls()
            children = _top_level_names(instance)
            n_params = sum(p.numel() for p in instance.parameters())
            print(f"  │  Sub-modules  : {children}")
            print(f"  │  Parameters   : {n_params:,}  ({n_params/1e6:.1f} M)")
        except Exception as exc:
            print(f"  │  (could not instantiate: {exc})")
        print(f"  └─")

    # ── Load and inspect checkpoint ───────────────────────────────────
    print(f"\n  Checkpoint: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print("  (file not found — skipping checkpoint analysis)")
        sys.exit(0)

    size_mb = os.path.getsize(args.checkpoint) / 1e6
    print(f"  File size : {size_mb:.0f} MB")

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw:
                raw = raw[key]
                break

    n_raw = len(raw) if isinstance(raw, dict) else "N/A"
    print(f"  State dict: {n_raw} keys")

    if isinstance(raw, dict):
        # Show prefix status
        first = next(iter(raw))
        if first.startswith("_orig_mod."):
            print(f"  Prefix    : '_orig_mod.'  (torch.compile artefact — will be stripped)")
        else:
            print(f"  Prefix    : none  (clean state dict)")

        stripped = _strip_orig_mod(raw)
        top_keys = sorted({k.split(".")[0] for k in stripped})
        print(f"  Top-level sub-modules in checkpoint: {top_keys}")

        # Run discovery
        print()
        print("  Running _discover_model_class …")
        try:
            cls = _discover_model_class(mod, stripped)
            print(f"\n  ✓  Would select: '{cls.__name__}'")
        except RuntimeError as exc:
            print(f"\n  ✗  Discovery failed:\n  {exc}")

    print()


if __name__ == "__main__":
    main()
    