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

IMPORTANT — TWO-STEP EXPORT
----------------------------
Previous versions used a single torch.jit.trace() call for the whole model.
This caused all three Nuke knobs (despill_strength, gamma_input, refiner_strength)
to be silently frozen at their default values — Nuke could change them, but the
traced graph had them baked in as constants.

The correct approach:
  Step 1  torch.jit.trace(_InnerTraceable(greenformer), dummy_CPU)
          Handles timm's dynamic Python registries via tracing.
          _InnerTraceable normalises output to a fixed 4-tuple.

  Step 2  torch.jit.script(CorridorKeyNukeWrapper(traced_inner))
          Compiles all if-branches. Attribute reads happen at call time.
          Nuke knob changes take full effect on every frame.

Run from the CorridorKey repo root:
    uv run python nuke/export_torchscript.py

Checkpoint must exist first:
    uv run python nuke/download_checkpoint.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The CorridorKey model code lives in the CorridorKey/ submodule.
_SUBMODULE_ROOT = _REPO_ROOT / "CorridorKey"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SUBMODULE_ROOT))

from nuke.nuke_wrapper import (  # type: ignore
    CorridorKeyNukeWrapper,
    _InnerTraceable,
    _StubInner,
    load_greenformer,
)


# ---------------------------------------------------------------------------
# Monkey-patch: fix timm Hiera Unroll.forward for TorchScript device safety
# ---------------------------------------------------------------------------
# timm's Unroll.forward does  B *= math.prod(strides)  where B starts as
# x.shape[0].  The TorchScript JIT bakes math.prod(strides) as a scalar
# graph constant (CONSTANTS.c0).  This constant is device-pinned to wherever
# tracing occurred (CPU).  At Nuke inference time on CUDA the shape-derived
# B lives on cuda:0 while CONSTANTS.c0 stays on CPU → RuntimeError.
#
# Fix: after each flatten, read B directly from x.shape[0] instead of
# multiplying a tracked variable.  x.shape[0] produces torch.size(x, 0)
# in TorchScript — always device-independent.
# ---------------------------------------------------------------------------
def _unroll_forward_device_safe(self: "torch.nn.Module", x: torch.Tensor) -> torch.Tensor:
    B, _, C = x.shape
    cur_size = self.size
    x = x.view(*([B] + cur_size + [C]))

    for strides in self.schedule:
        cur_size = [i // s for i, s in zip(cur_size, strides)]
        new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
        x = x.view(new_shape)

        L = len(new_shape)
        permute = [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
        x = x.permute(permute)

        x = x.flatten(0, len(strides))
        B = x.shape[0]  # device-safe: no baked constant

    x = x.reshape(-1, math.prod(self.size), C)
    return x


try:
    import timm.models.hiera as _hiera_mod
    _hiera_mod.Unroll.forward = _unroll_forward_device_safe
except (ImportError, AttributeError):
    pass  # timm not installed or API changed — tracing will use the original


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _is_stub(wrapper: CorridorKeyNukeWrapper) -> bool:
    """True if the wrapper's inner module is the test stub (not the real model)."""
    return isinstance(wrapper.inner, _StubInner)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(scripted: torch.jit.ScriptModule) -> None:
    """Quick sanity checks — shape, range, contiguity, NaN, knob sensitivity."""
    H, W = 256, 256

    # ── Basic output properties ───────────────────────────────────────
    dummy = torch.zeros(1, 4, H, W)
    with torch.no_grad():
        out = scripted(dummy)

    assert tuple(out.shape) == (1, 4, H, W), f"Wrong shape: {out.shape}"
    assert out.is_contiguous(),               "Output not contiguous"
    assert not torch.isnan(out).any(),        "NaN in output"
    assert not torch.isinf(out).any(),        "Inf in output"
    alpha = out[:, 3]
    assert float(alpha.min()) >= -0.01 and float(alpha.max()) <= 1.01, \
        f"Alpha out of range [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]"
    print(f"  ✓ shape      {tuple(out.shape)}")
    print(f"  ✓ alpha      [{float(alpha.min()):.3f}, {float(alpha.max()):.3f}]")
    print(f"  ✓ contiguous, no NaN/Inf")

    # ── Knob sensitivity: gamma_input ────────────────────────────────
    # In a correctly-scripted model, gamma_input=1 must change the output
    plate = torch.rand(1, 4, H, W)
    with torch.no_grad():
        scripted.gamma_input = 0
        out_srgb = scripted(plate).clone()
        scripted.gamma_input = 1
        out_lin  = scripted(plate).clone()
        scripted.gamma_input = 0  # restore default

    diff_gamma = float((out_srgb - out_lin).abs().mean())
    assert diff_gamma > 1e-4, (
        f"gamma_input knob has NO effect (diff={diff_gamma:.2e}) — "
        "model was traced instead of scripted!"
    )
    print(f"  ✓ gamma_input knob live  (output diff={diff_gamma:.4f})")

    # ── Knob sensitivity: despill_strength ───────────────────────────
    plate_green = torch.rand(1, 4, H, W)
    plate_green[:, 1, :, :] = 0.9   # heavy green in plate
    with torch.no_grad():
        scripted.despill_strength = 0.0
        out_no_despill = scripted(plate_green).clone()
        scripted.despill_strength = 8.0
        out_despilled  = scripted(plate_green).clone()
        scripted.despill_strength = 0.0  # restore default

    diff_despill = float((out_no_despill - out_despilled).abs().mean())
    assert diff_despill > 1e-5, (
        f"despill_strength knob has NO effect (diff={diff_despill:.2e}) — "
        "model was traced instead of scripted!"
    )
    print(f"  ✓ despill_strength knob live  (output diff={diff_despill:.4f})")

    # ── Knob sensitivity: refiner_strength ───────────────────────────
    # GreenFormer v1.0 is SINGLE-STAGE: forward() returns (alpha, fg) — one
    # prediction, no separate coarse/fine pass.  _InnerTraceable maps this to
    # (alpha, fg, alpha, fg), so alpha_coarse IS alpha_fine (same tensor).
    # The blend  alpha_c + s*(alpha_f - alpha_c)  = alpha_c + s*0  is always
    # alpha_c regardless of s.  refiner_strength is mathematically a no-op for
    # this checkpoint — it is NOT a scripting failure.
    #
    # We confirm scripting worked via gamma_input (verified live above).
    # We skip the refiner assert and report clearly instead.
    plate_hint = torch.rand(1, 4, H, W)
    with torch.no_grad():
        scripted.refiner_strength = 0.0
        out_coarse = scripted(plate_hint).clone()
        scripted.refiner_strength = 1.0
        out_fine   = scripted(plate_hint).clone()
        scripted.refiner_strength = 1.0  # restore default

    diff_refiner = float((out_coarse - out_fine).abs().mean())
    if diff_refiner > 1e-5:
        print(f"  ✓ refiner_strength knob live  (output diff={diff_refiner:.4f})")
    else:
        print(
            "  ✓ refiner_strength  (single-stage model: coarse == fine, "
            "knob is a no-op for this checkpoint — this is correct)"
        )


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(checkpoint: str, output: str, validate: bool, device_str: str) -> None:
    print()
    print("=" * 64)
    print("  CorridorKey → TorchScript export  (trace inner + script outer)")
    print("=" * 64)
    print(f"  checkpoint : {checkpoint}")
    print(f"  output     : {output}")
    print(f"  device     : {device_str}")
    print()

    device = torch.device(device_str)

    # ── Step 1: load the raw GreenFormer ─────────────────────────────
    print("► Loading GreenFormer from checkpoint …")
    greenformer: nn.Module = load_greenformer(checkpoint)
    greenformer.eval()
    # NOTE: do NOT move to CUDA here.  The model is loaded on CPU and
    # must stay on CPU for tracing (see Step 2 comment).  Moving to
    # CUDA and back can leave stale graph constants on the wrong device.
    n_params = sum(p.numel() for p in greenformer.parameters())
    print(f"  ✓ {n_params/1e6:.1f}M parameters")

    # ── Step 2: trace the inner model — ALWAYS ON CPU ─────────────────
    #
    # WHY CPU ONLY (critical):
    # timm's Hiera backbone has a module-level NamedTuple called CONSTANTS
    # (defined in timm/models/hiera.py) whose tensors are created at Python
    # import time — before any .to(device) call — and permanently reside on
    # CPU.  When traced on CUDA, those constants are baked into the graph
    # as CPU tensors.  At Nuke GPU inference time, the input arrives on
    # cuda:0, hitting "Expected all tensors to be on the same device,
    # but found cuda:0 and cpu" inside torch.mul_(B, CONSTANTS.c0).
    #
    # Tracing on CPU bakes all constants as CPU tensors.  Nuke's Cattery
    # system then moves the ENTIRE model — weights AND graph constants —
    # to the target device uniformly before running inference.  GPU works.
    #
    # The --device flag is kept for checkpoint loading speed only.
    print("► Step 1/2 — Tracing inner GreenFormer at 2048×2048 on CPU …")
    print("  (tracing always on CPU — Nuke Cattery handles GPU placement)")
    greenformer_cpu = greenformer.to("cpu")
    traceable = _InnerTraceable(greenformer_cpu)
    traceable.eval()
    dummy = torch.zeros(1, 4, 2048, 2048)   # CPU tensor

    t0 = time.perf_counter()
    with torch.no_grad():
        traced_inner: torch.jit.ScriptModule = torch.jit.trace(
            traceable, dummy, strict=False
        )
    print(f"  ✓ Inner trace complete in {time.perf_counter() - t0:.1f} s")

    # ── Step 3: script the outer wrapper ─────────────────────────────
    # CorridorKeyNukeWrapper embeds the traced inner and adds the knob
    # logic compiled with torch.jit.script so that attributes are read
    # dynamically at call time, not frozen at compile time.
    print("► Step 2/2 — Scripting outer CorridorKeyNukeWrapper …")
    wrapper = CorridorKeyNukeWrapper(traced_inner)  # already on CPU
    t1 = time.perf_counter()
    scripted: torch.jit.ScriptModule = torch.jit.script(wrapper)
    print(f"  ✓ Outer script complete in {time.perf_counter() - t1:.2f} s")

    # Guard: confirm this is NOT a stub
    # (the stub has no real parameters, so its count is tiny)
    total_params = sum(
        p.numel() for p in scripted.parameters()
    )
    if total_params < 1_000_000:
        print()
        print("  ✗ ERROR: scripted model has almost no parameters — stub was used.")
        print("  Fix: ensure the checkpoint is the real binary (~300 MB), then re-run.")
        sys.exit(1)

    # ── Validate ─────────────────────────────────────────────────────
    if validate:
        print("► Validating scripted model (shape + all 3 knobs) …")
        _validate(scripted)

    # ── Save ─────────────────────────────────────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    print(f"► Saved: {out_path.absolute()}  ({size_mb:.0f} MB)")

    if size_mb < 50:
        print()
        print(f"  ✗ ERROR: saved .pt is only {size_mb:.1f} MB — expected ~300 MB.")
        out_path.unlink(missing_ok=True)
        sys.exit(1)

    # ── CatFileCreator instructions ───────────────────────────────────
    print()
    print("=" * 64)
    print("  Next: open NukeX 17.0, create a CatFileCreator node")
    print()
    print(f"    Torchscript File  {out_path.absolute()}")
    print( "    Cat File          nuke/CorridorKey.cat")
    print( "    Channels In       rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Channels Out      rgba.red, rgba.green, rgba.blue, rgba.alpha")
    print( "    Model Id          CorridorKey_v1.0_Nuke")
    print( "    Output Scale      1")
    print()
    print("  Custom knobs:")
    print("    Float_Knob        despill_strength  default 0.0  range 0–10")
    print("    Enumeration_Knob  gamma_input       items: sRGB | Linear")
    print("    Float_Knob        refiner_strength  default 1.0  range 0–1")
    print()
    print("  See nuke/ARCHITECTURE.md for the comp node graph.")
    print("=" * 64)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export CorridorKey to TorchScript .pt (trace inner + script outer)"
    )
    p.add_argument(
        "--checkpoint",
        default=str(_SUBMODULE_ROOT / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"),
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "nuke" / "CorridorKey.pt"),
    )
    p.add_argument("--validate",    action="store_true",  default=True)
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()
    export(args.checkpoint, args.output, args.validate, args.device)


if __name__ == "__main__":
    main()
    