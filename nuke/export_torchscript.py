#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
###############################################################################
"""
export_torchscript.py
=====================
Exports CorridorKey v1.0 to a TorchScript .pt file ready for Nuke CatFileCreator.

Run from the CorridorKey repo root:

    uv run python nuke/export_torchscript.py

REQUIRED: download the real checkpoint first (once):

    uv run python nuke/download_checkpoint.py

The script will refuse to run if the checkpoint is missing, is a Git LFS
pointer, or if model loading falls back to the stub for any reason.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Ensure the repo root is importable regardless of CWD
# _INNER_ROOT is the Python package root (contains CorridorKeyModule, etc.)
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_INNER_ROOT = _REPO_ROOT / "CorridorKey"
sys.path.insert(0, str(_REPO_ROOT))

from nuke.nuke_wrapper import CorridorKeyNukeWrapper, _StubInner  # type: ignore


# ---------------------------------------------------------------------------
# Guard: detect stub
# ---------------------------------------------------------------------------

def _is_stub(model: CorridorKeyNukeWrapper) -> bool:
    """Return True if the model is using the test stub instead of the real network."""
    inner = model.model.inner
    return isinstance(inner, _StubInner)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(traced: torch.jit.ScriptModule) -> None:
    """Quick sanity checks — shape, range, contiguity, NaN, hint response."""
    H, W = 256, 256
    dummy = torch.zeros(1, 4, H, W)
    with torch.no_grad():
        out = traced(dummy)

    assert out.shape == (1, 4, H, W), f"Wrong shape: {out.shape}"
    assert out.is_contiguous(),        "Output not contiguous"

    alpha = out[:, 3]
    assert float(alpha.min()) >= -0.01 and float(alpha.max()) <= 1.01, \
        f"Alpha out of range: [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    # White hint → alpha > 0
    xw = torch.rand(1, 4, H, W)
    xw[:, 3] = 1.0
    with torch.no_grad():
        ow = traced(xw)
    assert float(ow[:, 3].mean()) > 0.0, "Alpha all-zero with white hint"

    # Black hint → alpha < 1
    xb = torch.rand(1, 4, H, W)
    xb[:, 3] = 0.0
    with torch.no_grad():
        ob = traced(xb)
    assert float(ob[:, 3].mean()) < 0.95, "Alpha all-one with black hint"

    print(f"  ✓ shape      {tuple(out.shape)}")
    print(f"  ✓ alpha      [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]")
    print(f"  ✓ contiguous")
    print(f"  ✓ no NaN / Inf")
    print(f"  ✓ hint response")


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(checkpoint: str, output: str, validate: bool, device_str: str) -> None:
    print()
    print("=" * 62)
    print("  CorridorKey → TorchScript export")
    print("=" * 62)
    print(f"  checkpoint : {checkpoint}")
    print(f"  output     : {output}")
    print(f"  device     : {device_str}")
    print()

    device = torch.device(device_str)

    # ── 1. Load wrapper ───────────────────────────────────────────────
    print("► Loading CorridorKeyNukeWrapper …")
    # _load_model raises with a clear message if the checkpoint is
    # missing, is a Git LFS pointer, or fails to deserialise.
    model = CorridorKeyNukeWrapper(checkpoint_path=checkpoint)
    model.eval()
    model.to(device)

    # ── 2. Abort if stub was loaded ───────────────────────────────────
    # This catches the exact failure mode from the user's console:
    # checkpoint is a Git LFS pointer → _load_model should have raised,
    # but if something bypassed it, we catch it here as a final guard.
    if _is_stub(model):
        print()
        print("  ✗ ERROR: stub model loaded instead of the real network.")
        print()
        print("  This means the checkpoint either does not exist, is a")
        print("  Git LFS pointer (not the real binary), or failed to load.")
        print()
        print("  Fix: run  uv run python nuke/download_checkpoint.py")
        print()
        sys.exit(1)

    # ── 3. Report parameter count ────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Real model loaded  ({total_params / 1e6:.1f} M parameters)")

    # ── 4. Trace ──────────────────────────────────────────────────────
    print("► Tracing at 2048×2048 (strict=False) …")
    dummy = torch.zeros(1, 4, 2048, 2048, device=device)

    if device_str == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ Trace complete in {elapsed:.1f} s")

    if device_str == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  ✓ Peak VRAM {peak_gb:.2f} GB")
        if peak_gb < 5.0:
            print("  ⚠ Peak VRAM unexpectedly low — confirm real model was traced.")

    # ── 5. Validate ───────────────────────────────────────────────────
    if validate:
        print("► Validating traced model …")
        traced_cpu = traced.to("cpu")
        _validate(traced_cpu)

    # ── 6. Save ───────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"► Saved: {out_path.absolute()}  ({size_mb:.0f} MB)")

    # ── 7. Guard: reject suspiciously small output ────────────────────
    if size_mb < 50:
        print()
        print(f"  ✗ ERROR: saved .pt is only {size_mb:.1f} MB — expected ~300 MB.")
        print("  The stub model was somehow traced instead of the real network.")
        print("  Delete the .pt and re-run after fixing the checkpoint.")
        out_path.unlink(missing_ok=True)
        sys.exit(1)

    # ── 8. CatFileCreator instructions ────────────────────────────────
    print()
    print("=" * 62)
    print("  Next: open NukeX 17.0, create a CatFileCreator node")
    print()
    print(f"    Torchscript File  {out_path.absolute()}")
    print( "    Cat File          Export/CorridorKey.cat")
    print( "    Channels In       rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Channels Out      rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Model Id          CorridorKey_v1.0_Nuke")
    print( "    Output Scale      1")
    print()
    print("  Custom knobs (drag onto CatFileCreator panel):")
    print("    Float_Knob        despill_strength  default 0.0  range 0–10")
    print("    Enumeration_Knob  gamma_input       items: sRGB | Linear")
    print("    Float_Knob        refiner_strength  default 1.0  range 0–1")
    print()
    print("  See nuke/ARCHITECTURE.md for the compositing node graph.")
    print("=" * 62)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Export CorridorKey to TorchScript .pt")
    p.add_argument(
        "--checkpoint",
        default=str(_INNER_ROOT / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"),
        help="Path to CorridorKey.pth  (~300 MB real weights)",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "Export" / "CorridorKey.pt"),
        help="Output path for the TorchScript .pt file",
    )
    p.add_argument(
        "--validate", action="store_true", default=True,
        help="Run post-trace sanity checks (default: on)",
    )
    p.add_argument(
        "--no-validate", dest="validate", action="store_false",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Trace device",
    )
    args = p.parse_args()

    export(
        checkpoint=args.checkpoint,
        output=args.output,
        validate=args.validate,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()