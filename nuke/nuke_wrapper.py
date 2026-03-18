###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
###############################################################################
"""
nuke_wrapper.py
===============
TorchScript-compatible Nuke Cattery wrapper for CorridorKey v1.0.

WHY TRACE + SCRIPT INSTEAD OF TRACE ALONE
------------------------------------------
torch.jit.trace() records a concrete execution graph from a single forward
pass.  Any Python 'if' branch that evaluates to False at trace time is simply
not recorded — it is permanently absent from the graph.  This means:

  - gamma_input    defaults to 0 → the sRGB encoding branch is NEVER traced
  - despill_strength defaults to 0 → the despill branch is NEVER traced
  - refiner_strength defaults to 1.0 → baked in as a constant 1.0

Result: changing any knob in Nuke has no effect on the output.

CORRECT ARCHITECTURE (two-step export)
---------------------------------------
  Step 1 — TRACE the inner GreenFormer only.
            torch.jit.trace handles timm's dynamic Python registries.
            Output: a ScriptModule with fixed forward() semantics.

  Step 2 — SCRIPT the outer CorridorKeyNukeWrapper.
            torch.jit.script compiles the wrapper including all if-branches.
            Attribute reads (self.gamma_input etc.) happen at call time, not
            at compile time → Nuke knobs work correctly.

NUKE .cat I/O
-------------
  Input  [1, 4, H, W]  rgba.red/.green/.blue = plate,  rgba.alpha = hint
  Output [1, 4, H, W]  rgba.red/.green/.blue = sRGB FG, rgba.alpha = linear alpha

CUSTOM KNOBS  (attribute names must match CatFileCreator "Name" exactly)
  despill_strength  float  0-10   green despill intensity
  gamma_input       int    0/1    0 = sRGB plate,  1 = linear EXR plate
  refiner_strength  float  0-1    0 = coarse only, 1 = full CNN refinement
"""

import importlib
import inspect
import math
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# ── Loading utilities (used by export_torchscript.py at export time only) ──
# ===========================================================================

def _strip_orig_mod(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove '_orig_mod.' prefix added by torch.compile()."""
    PREFIX = "_orig_mod."
    if not state or not next(iter(state)).startswith(PREFIX):
        return state
    stripped = {k[len(PREFIX):]: v for k, v in state.items()}
    print(
        f"  [CorridorKey] Stripped '{PREFIX}' prefix from {len(stripped)} keys  "
        "(checkpoint saved from torch.compile())"
    )
    return stripped


def _top_level_submodule_names(net: nn.Module) -> set:
    return {name for name, _ in net.named_children()}


def _read_patch_stride(instance: nn.Module) -> Optional[int]:
    """Read the actual Conv stride from patch_embed — NOT the kernel size."""
    try:
        conv = instance.encoder.model.patch_embed.proj
        if isinstance(conv, nn.Conv2d):
            s = conv.stride
            return s[0] if isinstance(s, (tuple, list)) else int(s)
    except AttributeError:
        pass
    for _, mod in instance.named_modules():
        if isinstance(mod, nn.Conv2d):
            attrs = [a for a in dir(mod) if 'patch' in a.lower()]
            if attrs:
                s = mod.stride
                return s[0] if isinstance(s, (tuple, list)) else int(s)
    return None


def _pos_embed_num_patches(instance: nn.Module) -> Optional[int]:
    try:
        return instance.encoder.model.pos_embed.shape[1]
    except AttributeError:
        for name, p in instance.named_parameters():
            if "pos_embed" in name:
                return p.shape[1]
    return None


def _discover_model_class(
    module: types.ModuleType,
    stripped_state: Dict[str, torch.Tensor],
) -> Tuple[type, nn.Module]:
    """
    Find the correct model class and a properly-instantiated instance.
    Infers img_size from checkpoint pos_embed by reading the STRIDE
    (not kernel_size) from a default-instantiated model.
    """
    required_names = {k.split(".")[0] for k in stripped_state}
    print(f"  [CorridorKey] Required sub-modules: {sorted(required_names)}")

    target_patches: Optional[int] = None
    for k, v in stripped_state.items():
        if "pos_embed" in k:
            target_patches = v.shape[1]
            break

    candidates: List[Tuple[str, type]] = [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if not name.startswith("_")
        and isinstance(obj, type)
        and issubclass(obj, nn.Module)
        and getattr(obj, "__module__", "") == module.__name__
    ]

    if not candidates:
        raise RuntimeError(
            f"No public nn.Module subclasses found in {module.__file__}."
        )

    best_cls:      Optional[type]      = None
    best_instance: Optional[nn.Module] = None
    best_score     = -1
    best_params    = -1

    for cls_name, cls in candidates:
        try:
            sig = inspect.signature(cls.__init__)
            required_args = [
                n for n, p in sig.parameters.items()
                if n != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            accepts_img_size = "img_size" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            required_args    = []
            accepts_img_size = False

        if required_args:
            continue

        img_sizes_to_try: List[Optional[int]] = [None]

        if target_patches is not None and accepts_img_size:
            target_grid = math.isqrt(target_patches)
            # Strategy: instantiate with defaults, read actual stride
            try:
                default_inst = cls()
                stride = _read_patch_stride(default_inst)
                del default_inst
                if stride and stride > 0:
                    computed = target_grid * stride
                    img_sizes_to_try.insert(0, computed)
                    print(
                        f"  [CorridorKey] Inferred img_size={computed} "
                        f"from stride={stride} "
                        f"(num_patches={target_patches}={target_grid}²)"
                    )
            except Exception:
                pass
            # Fallback: common strides
            for st in (4, 2, 8, 1, 16, 3, 7, 14, 32):
                gs = target_grid * st
                if gs not in img_sizes_to_try:
                    img_sizes_to_try.append(gs)

        for img_size in img_sizes_to_try:
            kwargs: Dict[str, Any] = {}
            if img_size is not None and accepts_img_size:
                kwargs["img_size"] = img_size

            try:
                inst = cls(**kwargs)
                inst.eval()
            except Exception:
                continue

            if not required_names.issubset(_top_level_submodule_names(inst)):
                del inst
                continue

            if target_patches is not None:
                actual = _pos_embed_num_patches(inst)
                if actual is not None and actual != target_patches:
                    del inst
                    continue

            try:
                result   = inst.load_state_dict(stripped_state, strict=False)
                n_bad    = len(result.missing_keys) + len(result.unexpected_keys)
                score    = len(stripped_state) - n_bad
                n_params = sum(p.numel() for p in inst.parameters())
            except Exception:
                score    = 0
                n_params = 0

            if score > best_score or (score == best_score and n_params > best_params):
                best_score    = score
                best_params   = n_params
                best_cls      = cls
                best_instance = inst
                print(
                    f"  [CorridorKey] '{cls_name}' img_size={img_size} "
                    f"→ score {score}/{len(stripped_state)}, "
                    f"{n_params/1e6:.1f}M params"
                )
            else:
                del inst
            break

    if best_cls is None or best_instance is None:
        raise RuntimeError(
            f"No class matched sub-modules {sorted(required_names)} in {module.__file__}.\n"
            f"Classes found: {[n for n, _ in candidates]}\n"
            "Run  uv run python nuke/inspect_model.py  for diagnostics."
        )

    print(
        f"  [CorridorKey] Selected '{best_cls.__name__}'  "
        f"(score {best_score}/{len(stripped_state)}, "
        f"{best_params/1e6:.1f}M params)"
    )
    return best_cls, best_instance


def load_greenformer(checkpoint_path: str) -> nn.Module:
    """
    Load the raw GreenFormer nn.Module with weights applied.
    Called by export_torchscript.py — not used at Nuke runtime.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"\nCheckpoint not found: {checkpoint_path}\n"
            "Download: uv run python nuke/download_checkpoint.py\n"
        )

    size_bytes = os.path.getsize(checkpoint_path)
    if size_bytes < 1_000_000:
        lfs_hint = ""
        try:
            with open(checkpoint_path, "rb") as f:
                h = f.read(200).decode("utf-8", errors="replace")
            if "git-lfs" in h or "oid sha256" in h:
                lfs_hint = "\n(Git LFS pointer — re-download with nuke/download_checkpoint.py)\n"
        except Exception:
            pass
        raise ValueError(
            f"\nCheckpoint is only {size_bytes:,} bytes (expected ~300 MB).{lfs_hint}"
        )

    # nuke/ sits one level above the repo root *or* inside the CorridorKey
    # submodule. Try both so the same file works from either location.
    _nuke_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        os.path.abspath(os.path.join(_nuke_dir, "..")),             # .../CorridorKey/
        os.path.abspath(os.path.join(_nuke_dir, "..", "CorridorKey")),  # outer repo root → CorridorKey/
    ]
    for _candidate in _candidates:
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)

    try:
        _mt_mod = importlib.import_module("CorridorKeyModule.core.model_transformer")
    except ImportError as exc:
        raise ImportError(
            f"\nCould not import CorridorKeyModule.core.model_transformer: {exc}\n"
            "Run from repo root:  uv sync\n"
        ) from exc

    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"\ntorch.load failed: {exc}\n") from exc

    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in raw:
                raw = raw[key]
                break

    raw = _strip_orig_mod(raw)

    _ModelClass, net = _discover_model_class(_mt_mod, raw)

    try:
        net.load_state_dict(raw, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"\nload_state_dict failed for '{_ModelClass.__name__}':\n{exc}\n"
            "Run  uv run python nuke/inspect_model.py  for diagnostics.\n"
        ) from exc

    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"  [CorridorKey] Loaded {size_bytes/1e6:.0f} MB checkpoint "
        f"→ '{_ModelClass.__name__}'  ({n_params/1e6:.1f}M params)"
    )

    # Detect single-stage vs two-stage model by doing a tiny probe forward
    # and checking whether the output length is 2 or 4.
    # Single-stage (len==2): refiner_strength is a no-op (coarse == fine).
    # Two-stage  (len==4): refiner_strength blends coarse → refined output.
    try:
        with torch.no_grad():
            probe_out = net(torch.zeros(1, 4, 64, 64))
        n_outputs = len(probe_out) if isinstance(probe_out, (list, tuple)) else 0
        if n_outputs == 2:
            print(
                "  [CorridorKey] Single-stage model detected "
                "(forward returns 2 outputs). "
                "refiner_strength knob will have no effect in Nuke — "
                "this is correct for this checkpoint."
            )
        elif n_outputs == 4:
            print(
                "  [CorridorKey] Two-stage model detected "
                "(forward returns 4 outputs). "
                "refiner_strength knob is fully live."
            )
    except Exception:
        pass  # probe is optional — never block the export

    return net


# ===========================================================================
# ── _InnerTraceable: wraps GreenFormer for torch.jit.trace ─────────────────
#
# This class is TRACED (not scripted) to handle timm's dynamic Python
# registries.  It normalises GreenFormer's output to a fixed 4-tuple so
# that the outer scripted wrapper has a known return type to unpack.
# ===========================================================================

class _InnerTraceable(nn.Module):
    """
    Wraps the raw GreenFormer and normalises its output to:
      Tuple[alpha_coarse, fg_coarse, alpha_fine, fg_fine]

    This class is intended to be exported via torch.jit.trace(), NOT script().
    The isinstance() checks in forward() run at trace time (fine for tracing)
    and the resulting graph records only the tensor ops that were executed.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(x)

        if isinstance(out, dict):
            if "alpha_coarse" in out:
                return (
                    out["alpha_coarse"], out["fg_coarse"],
                    out.get("alpha_fine", out["alpha_coarse"]),
                    out.get("fg_fine",    out["fg_coarse"]),
                )
            if "alpha" in out and "fg" in out:
                return (out["alpha"], out["fg"], out["alpha"], out["fg"])
            vals = list(out.values())
            return (vals[0], vals[1], vals[0], vals[1])

        if isinstance(out, (list, tuple)) and len(out) == 4:
            return (out[0], out[1], out[2], out[3])
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return (out[0], out[1], out[0], out[1])

        raise ValueError(f"Unexpected GreenFormer output type: {type(out)}")


# ===========================================================================
# ── Stub: for tests / CI when no checkpoint is provided ────────────────────
# ===========================================================================

class _StubInner(nn.Module):
    """Correct shape/range, meaningless values.  Tests and CI only."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha_head = nn.Conv2d(4, 1, 1)
        self.fg_head    = nn.Conv2d(4, 3, 1)
        nn.init.constant_(self.alpha_head.bias, -2.0)
        nn.init.constant_(self.fg_head.bias,     0.5)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a  = torch.sigmoid(self.alpha_head(x))
        fg = torch.sigmoid(self.fg_head(x))
        return (a, fg,
                torch.clamp(a  + 0.01  * x[:, 3:4], 0.0, 1.0),
                torch.clamp(fg + 0.005 * x[:, :3],  0.0, 1.0))


# ===========================================================================
# ── CorridorKeyNukeWrapper: the SCRIPTABLE outer wrapper ───────────────────
#
# This class is exported via torch.jit.SCRIPT (not trace).
# In script mode, self.gamma_input / self.despill_strength / self.refiner_strength
# are read dynamically at each forward() call, so Nuke knob changes take effect.
# ===========================================================================

class CorridorKeyNukeWrapper(nn.Module):
    """
    Scriptable Nuke Cattery wrapper.

    Accepts a pre-traced inner module (from _InnerTraceable or _StubInner).
    The knob attributes are read at call time — NOT frozen at compile time.

    Export via:
        traced_inner = torch.jit.trace(_InnerTraceable(greenformer), dummy)
        wrapper      = CorridorKeyNukeWrapper(traced_inner)
        scripted     = torch.jit.script(wrapper)
        scripted.save("CorridorKey.pt")
    """

    # TorchScript requires explicit type annotations for all persistent attrs
    despill_strength: float
    gamma_input:      int
    refiner_strength: float

    _MODEL_H: int
    _MODEL_W: int

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

        # Knob defaults — names MUST match CatFileCreator "Name" fields exactly
        self.despill_strength = 0.0
        self.gamma_input      = 0
        self.refiner_strength = 1.0

        # Native model resolution (must be int constants for TorchScript)
        self._MODEL_H = 2048
        self._MODEL_W = 2048

    # ------------------------------------------------------------------
    # Color math — TorchScript-safe (pure torch ops, no numpy, no cv2)
    # ------------------------------------------------------------------

    def _linear_to_srgb(self, x: torch.Tensor) -> torch.Tensor:
        """IEC 61966-2-1: linear light → sRGB gamma encoding."""
        x  = torch.clamp(x, 0.0, 1.0)
        lo = x * 12.92
        hi = 1.055 * torch.pow(torch.clamp(x, min=1e-9), 1.0 / 2.4) - 0.055
        return torch.clamp(torch.where(x <= 0.0031308, lo, hi), 0.0, 1.0)

    def _despill_green(self, fg: torch.Tensor) -> torch.Tensor:
        """Average green despill.  Only the G channel is modified."""
        r = fg[:, 0:1, :, :]
        g = fg[:, 1:2, :, :]
        b = fg[:, 2:3, :, :]
        neutral = (r + b) * 0.5
        blend   = torch.clamp(
            torch.tensor(self.despill_strength / 10.0,
                         dtype=fg.dtype, device=fg.device),
            0.0, 1.0,
        )
        g_out = g * (1.0 - blend) + torch.minimum(g, neutral) * blend
        return torch.cat([r, torch.clamp(g_out, 0.0, 1.0), b], dim=1)

    # ------------------------------------------------------------------
    # forward() — ALL branches are compiled and execute dynamically
    # ------------------------------------------------------------------

    def _soften_alpha(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Spatially soften an alpha matte by average-pooling then upsampling.
        Equivalent to a fast, separable box blur.  Kernel size = 9×9 at 2048px.
        Used by the refiner_strength knob to smooth chattery / high-frequency edges.
        """
        H: int = alpha.shape[2]
        W: int = alpha.shape[3]
        # Pool to 1/4 resolution then upsample — cheap spatial blur
        small = F.avg_pool2d(alpha, kernel_size=9, stride=1, padding=4)
        return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : [1, 4, H, W]  ch0-2 = plate RGB (sRGB),  ch3 = alpha hint
        out : [1, 4, H, W]  ch0-2 = straight sRGB FG,   ch3 = linear alpha

        All three knobs are read at call time (torch.jit.script behaviour):

          gamma_input      0 = sRGB plate (pass-through)
                           1 = linear EXR plate → encode to sRGB before model

          despill_strength 0 = off,  10 = full green-channel despill

          refiner_strength 1.0 = sharp model output  (default)
                           0.0 = spatially softened alpha
                           Useful for taming chattery / high-frequency matte edges.
                           NOTE: GreenFormer returns a single (alpha, fg) pair —
                           there is no separate coarse/fine output to blend.
                           refiner_strength therefore controls edge sharpness via
                           a lerp between the raw model alpha and a box-blurred
                           version of it.  FG colour is always at full sharpness.
        """
        dtype  = x.dtype
        H: int = x.shape[2]
        W: int = x.shape[3]

        plate = x[:, 0:3, :, :]
        hint  = x[:, 3:4, :, :]

        # ── gamma_input knob ─────────────────────────────────────────
        if self.gamma_input == 1:
            plate = self._linear_to_srgb(plate)

        # ── Resize to native model resolution ────────────────────────
        model_in = F.interpolate(
            torch.cat([plate, hint], dim=1),
            size=(self._MODEL_H, self._MODEL_W),
            mode="bilinear", align_corners=False,
        )

        # ── Neural network inference ──────────────────────────────────
        # GreenFormer is a single-stage model: it returns one (alpha, fg) pair.
        # _InnerTraceable maps any 2-tuple as (a, fg, a, fg) so alpha_c == alpha_f.
        # We use only alpha_c / fg_c (the actual model outputs).
        alpha_c, fg_c, _, _ = self.inner(model_in)

        alpha = torch.clamp(alpha_c, 0.0, 1.0)
        fg    = torch.clamp(fg_c,    0.0, 1.0)

        # ── Resize back to original plate resolution ─────────────────
        alpha = F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
        fg    = F.interpolate(fg,    size=(H, W), mode="bilinear", align_corners=False)

        # ── refiner_strength knob ────────────────────────────────────
        # 1.0 = sharp model output (default, no change)
        # 0.0 = softened alpha (smooths chattery / noisy matte edges)
        # Applied only to alpha — FG colour is always at full sharpness.
        if self.refiner_strength < 1.0:
            soft  = self._soften_alpha(alpha)
            s     = torch.clamp(
                torch.tensor(self.refiner_strength, dtype=alpha.dtype, device=alpha.device),
                0.0, 1.0,
            )
            alpha = s * alpha + (1.0 - s) * soft
            alpha = torch.clamp(alpha, 0.0, 1.0)

        # ── despill_strength knob ────────────────────────────────────
        if self.despill_strength > 0.0:
            fg = self._despill_green(fg)

        # ── Pack output: FG rgb (ch0-2) + alpha (ch3) ────────────────
        return torch.cat([fg, alpha], dim=1).contiguous().to(dtype)