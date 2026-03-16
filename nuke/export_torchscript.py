#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
###############################################################################
#!/usr/bin/env python3
"""
export_torchscript.py
=====================
Exports CorridorKey v1.0 to a TorchScript .pt file ready for Nuke CatFileCreator.
Compatible with Nuke 15.1+ and Nuke 17.0+.

Run from the CorridorKey repo root:

    uv run python nuke/export_torchscript.py

REQUIRED: download the real checkpoint first (once):

    uv run python nuke/download_checkpoint.py

Why this always traces on CPU
------------------------------
torch.jit.trace() records a concrete forward pass and bakes tensor constants
at trace time on whatever device the model lives on.

timm's hiera backbone stores shape-multiplier constants (CONSTANTS.c1) as
registered buffers. When traced on CUDA these become cuda:0 tensors baked
into the TorchScript graph. Nuke 15's CatFileCreator validates the model on
CPU → "Expected all tensors to be on the same device, cuda:0 and cpu" crash.

Tracing on CPU produces CPU-baked constants. The _patch_hiera_unroll() call
additionally converts those scalar constants to Python ints (no device at all)
so the same .pt runs correctly on CPU (Nuke 15) AND GPU (Nuke 17).
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# _INNER_ROOT is the Python package root (contains CorridorKeyModule, etc.)
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_INNER_ROOT = _REPO_ROOT / "CorridorKey"
sys.path.insert(0, str(_REPO_ROOT))

from nuke.nuke_wrapper import (  # type: ignore
    CorridorKeyNukeWrapper,
    _StubInner,
    _patch_hiera_unroll,
)


# ---------------------------------------------------------------------------
# Detect stub
# ---------------------------------------------------------------------------

def _is_stub(model: CorridorKeyNukeWrapper) -> bool:
    return isinstance(model.model.inner, _StubInner)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(traced: torch.jit.ScriptModule) -> None:
    """Shape, range, contiguity, NaN, hint-response checks on CPU."""
    H, W = 256, 256
    dummy = torch.zeros(1, 4, H, W)
    with torch.no_grad():
        out = traced(dummy)

    assert out.shape == (1, 4, H, W), f"Wrong shape: {out.shape}"
    assert out.is_contiguous(),        "Output not contiguous"

    alpha = out[:, 3]
    assert float(alpha.min()) >= -0.01 and float(alpha.max()) <= 1.01, \
        f"Alpha range [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    # White hint → some alpha
    xw = torch.rand(1, 4, H, W); xw[:, 3] = 1.0
    with torch.no_grad():
        ow = traced(xw)
    assert float(ow[:, 3].mean()) > 0.0, "Alpha all-zero with white hint"

    # Black hint → low alpha
    xb = torch.rand(1, 4, H, W); xb[:, 3] = 0.0
    with torch.no_grad():
        ob = traced(xb)
    assert float(ob[:, 3].mean()) < 0.95, "Alpha all-one with black hint"

    print(f"  ✓ shape    {tuple(out.shape)}")
    print(f"  ✓ alpha    [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]")
    print(f"  ✓ contiguous")
    print(f"  ✓ no NaN / Inf")
    print(f"  ✓ hint response")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(checkpoint: str, output: str, validate: bool) -> None:
    print()
    print("=" * 62)
    print("  CorridorKey → TorchScript  (Nuke 15.1 + Nuke 17.0 compatible)")
    print("=" * 62)
    print(f"  checkpoint  : {checkpoint}")
    print(f"  output      : {output}")
    print(f"  trace device: cpu  (always — see module docstring)")
    print()

    # ── 1. Apply hiera patch BEFORE loading model ─────────────────────────
    # This patches UnrollBlock.forward to convert scalar CONSTANTS to Python
    # ints, making the baked values device-agnostic (no cuda:0 tensor baked).
    patched = _patch_hiera_unroll()
    status  = "applied" if patched else "skipped (timm not found)"
    print(f"  hiera CONSTANTS patch: {status}")

    # ── 2. Load wrapper on CPU ────────────────────────────────────────────
    print("► Loading CorridorKeyNukeWrapper …")
    model = CorridorKeyNukeWrapper(checkpoint_path=checkpoint)
    model.eval()
    # IMPORTANT: do NOT call model.to(device) — trace on CPU

    if _is_stub(model):
        print()
        print("  ✗ ERROR: stub model loaded — real checkpoint not found or corrupt.")
        print("  Fix: uv run python nuke/download_checkpoint.py")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Real model  ({n_params/1e6:.1f} M parameters)")

    # ── 3. Trace on CPU ───────────────────────────────────────────────────
    print("► Tracing at 2048×2048 on CPU …")
    dummy = torch.zeros(1, 4, 2048, 2048)   # CPU tensor, no .cuda()

    t0 = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, strict=False)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ Trace complete in {elapsed:.1f} s")

    # ── 4. Validate ───────────────────────────────────────────────────────
    if validate:
        print("► Validating …")
        _validate(traced)

    # ── 5. Save ───────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"► Saved: {out_path.absolute()}  ({size_mb:.0f} MB)")

    if size_mb < 50:
        print()
        print(f"  ✗ ERROR: .pt is only {size_mb:.1f} MB — stub was traced.")
        out_path.unlink(missing_ok=True)
        sys.exit(1)

    # ── 6. CatFileCreator instructions ────────────────────────────────────
    print()
    print("=" * 62)
    print("  Next: open NukeX 15.1 or 17.0, import nuke/CatFileCreators.nk")
    print()
    print(f"    Torchscript File  {out_path.absolute()}")
    print( "    Cat File          Export/CorridorKey.cat")
    print( "    Channels In       rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Channels Out      rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Model Id          CorridorKey_v1.0_Nuke")
    print( "    Output Scale      1")
    print()
    print("  Custom knobs:")
    print("    Float_Knob        despill_strength  default 0.0  range 0–10")
    print("    Enumeration_Knob  gamma_input       sRGB | Linear")
    print("    Float_Knob        refiner_strength  default 1.0  range 0–1")
    print()
    print("  See nuke/README.md for the compositing node graph.")
    print("=" * 62)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export CorridorKey to TorchScript (Nuke 15 + 17 compatible)"
    )
    p.add_argument(
        "--checkpoint",
        default=str(_INNER_ROOT / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"),
        help="Path to CorridorKey.pth (~300 MB)",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "Export" / "CorridorKey.pt"),
        help="Output .pt path",
    )
    p.add_argument(
        "--validate", action="store_true", default=True,
        help="Post-trace sanity checks (default: on)",
    )
    p.add_argument(
        "--no-validate", dest="validate", action="store_false",
    )
    args = p.parse_args()

    export(
        checkpoint=args.checkpoint,
        output=args.output,
        validate=args.validate,
    )


if __name__ == "__main__":
    main()
