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
Compatible with Nuke 15.1+ and Nuke 17.0+.

Nuke .cat I/O
-------------
  Input  [1, 4, H, W]  rgba.red/.green/.blue = plate,  rgba.alpha = hint
  Output [1, 4, H, W]  rgba.red/.green/.blue = sRGB FG, rgba.alpha = linear alpha

Custom knobs (attribute names must match CatFileCreator "Name" exactly)
  despill_strength  float  0-10   green despill intensity
  gamma_input       int    0/1    0 = sRGB plate,  1 = linear EXR plate
  refiner_strength  float  0-1    0 = coarse only, 1 = full CNN refinement

Nuke 15 vs 17 compatibility — three bugs fixed
-----------------------------------------------

  BUG 1  CRASH  Device mismatch during torch.jit.trace()
  ───────────────────────────────────────────────────────
  Symptom: RuntimeError: Expected all tensors to be on the same device,
           cuda:0 and cpu — crash happens during export, not at inference.
  Cause:   When tracing on CUDA, timm's hiera.py has a registered buffer
           (CONSTANTS.c1) that lives on CUDA. Inside trace(), Python ints
           from x.shape[] are wrapped as 0-dim CPU tensors by the trace
           recorder. torch.mul_(cpu_tensor, cuda_tensor) → device mismatch.
  Fix A:   Always trace on CPU. CPU-baked CONSTANTS stay CPU at runtime.
  Fix B:   _patch_hiera_unroll() — called before tracing — monkey-patches
           UnrollBlock.forward to convert scalar CONSTANTS to Python ints
           (.item()) so they have NO device. The traced graph contains Python
           int literals, not device-specific tensors. Works on CPU or GPU.

  BUG 2  KNOB DISCONNECTED  "parameter not found" warnings in Nuke 15
  ─────────────────────────────────────────────────────────────────────
  Symptom: Cat file warning: The parameter despill_strength was not found.
  Cause:   Knob attributes were declared as instance annotations in __init__:
               self.despill_strength: float = 0.0
           TorchScript requires CLASS-LEVEL annotations to expose typed
           attributes. Without them, Nuke 15's CatFileCreator cannot find the
           attribute name, and the knob is disconnected from the computation.
  Fix:     Declare all three knobs at the CLASS level (not just __init__):
               class CorridorKeyNukeWrapper(nn.Module):
                   despill_strength: float   # ← class-level, visible to TS
                   gamma_input: int
                   refiner_strength: float
           With class-level annotations, TorchScript generates prim::GetAttr
           instructions (dynamic runtime reads) instead of baking values as
           CONSTANTS.c0, CONSTANTS.c1, etc.

  BUG 3  KNOB BAKED  if-branches freeze knob values at trace time
  ────────────────────────────────────────────────────────────────
  Symptom: Changing gamma_input or despill_strength in the Inference node
           has no effect on the output.
  Cause:   torch.jit.trace() records ONE execution path. Any branch not
           taken at trace time is permanently absent from the graph.
               if self.gamma_input == 1:   # traced with 0 → branch gone
           Even with class-level annotations, trace() sees `self.gamma_input`
           evaluate to 0 at trace time and bakes the NOT-taken branch out.
  Fix:     Replace every if-branch on knob values with continuous arithmetic.
           Both code paths are always recorded. The knob value controls a
           blend weight, which IS read dynamically via prim::GetAttr.
               # gamma_input=0 → plate * 1.0 + srgb * 0.0 = passthrough
               # gamma_input=1 → plate * 0.0 + srgb * 1.0 = encoded
               plate = plate * (1.0 - gi) + self._linear_to_srgb(plate) * gi
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# BUG 1 FIX — timm hiera CONSTANTS patch
# ---------------------------------------------------------------------------

def _patch_hiera_unroll() -> bool:
    """
    Monkey-patch timm's UnrollBlock (and RerollBlock) so that scalar
    CONSTANTS are converted to Python ints before torch.jit.trace() records
    them.

    Without this patch:
      torch.jit.trace(model.cuda(), dummy.cuda()) bakes CONSTANTS.c1 as a
      cuda:0 tensor. When Nuke 15 runs the model on CPU, the baked cuda
      constant meets a CPU input tensor → device mismatch crash.

    With this patch:
      CONSTANTS.c1.item() → Python int → no device → works on CPU or GPU.

    Returns True if timm is installed and the patch was applied.
    """
    try:
        import timm.models.hiera as _hiera
    except ImportError:
        return False

    if getattr(_hiera, "_ck_nuke_patched", False):
        return True

    _orig_mul_ = torch.Tensor.mul_

    def _device_safe_mul_(self_t: torch.Tensor, other: Any) -> torch.Tensor:
        # If `other` is a single-element tensor on a different device, convert
        # it to a Python scalar so the operation becomes device-agnostic.
        if isinstance(other, torch.Tensor):
            if other.numel() == 1 and other.device != self_t.device:
                other = other.item()
        return _orig_mul_(self_t, other)

    # Patch UnrollBlock.forward
    _orig_unroll = _hiera.UnrollBlock.forward

    def _unroll_patched(self_block, x: torch.Tensor) -> torch.Tensor:
        torch.Tensor.mul_ = _device_safe_mul_
        try:
            return _orig_unroll(self_block, x)
        finally:
            torch.Tensor.mul_ = _orig_mul_

    _hiera.UnrollBlock.forward = _unroll_patched

    # Patch RerollBlock.forward if it exists
    if hasattr(_hiera, "RerollBlock"):
        _orig_reroll = _hiera.RerollBlock.forward

        def _reroll_patched(self_block, x: torch.Tensor, size: Any) -> torch.Tensor:
            torch.Tensor.mul_ = _device_safe_mul_
            try:
                return _orig_reroll(self_block, x, size)
            finally:
                torch.Tensor.mul_ = _orig_mul_

        _hiera.RerollBlock.forward = _reroll_patched

    _hiera._ck_nuke_patched = True
    print("  [CorridorKey] timm hiera CONSTANTS patch applied (device-agnostic tracing)")
    return True


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model class discovery (unchanged — already robust)
# ---------------------------------------------------------------------------

def _top_level_submodule_names(net: nn.Module) -> set:
    return {name for name, _ in net.named_children()}


def _read_patch_stride(instance: nn.Module) -> Optional[int]:
    try:
        conv = instance.encoder.model.patch_embed.proj
        if isinstance(conv, nn.Conv2d):
            s = conv.stride
            return s[0] if isinstance(s, (tuple, list)) else int(s)
    except AttributeError:
        pass
    for name, mod in instance.named_modules():
        if isinstance(mod, nn.Conv2d) and "patch" in name.lower():
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
    required_names = {k.split(".")[0] for k in stripped_state}
    print(f"  [CorridorKey] Required sub-modules: {sorted(required_names)}")

    target_patches: Optional[int] = None
    for k, v in stripped_state.items():
        if "pos_embed" in k:
            target_patches = v.shape[1]
            break

    candidates: List[Tuple[str, type]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_"):
            continue
        if not (isinstance(obj, type) and issubclass(obj, nn.Module)):
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        candidates.append((name, obj))

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
            req = [
                n for n, p in sig.parameters.items()
                if n != "self"
                and p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if req:
                continue
        except (ValueError, TypeError):
            continue

        img_size_candidates: List[Optional[int]] = [None]
        try:
            sig_p = dict(inspect.signature(cls.__init__).parameters)
            accepts_img = "img_size" in sig_p or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_p.values()
            )
        except Exception:
            accepts_img = False

        if target_patches is not None and accepts_img:
            target_grid = math.isqrt(target_patches)
            try:
                di = cls()
                ps = _read_patch_stride(di)
                del di
                if ps and ps > 0:
                    computed = target_grid * ps
                    img_size_candidates.insert(0, computed)
                    print(
                        f"  [CorridorKey] Inferred img_size={computed} "
                        f"(stride={ps}, patches={target_patches}={target_grid}²)"
                    )
            except Exception:
                pass
            for sg in (4, 2, 8, 1, 16, 3, 7, 14, 32):
                gs = target_grid * sg
                if gs not in img_size_candidates:
                    img_size_candidates.append(gs)

        for img_size in img_size_candidates:
            kwargs: Dict[str, Any] = {}
            if img_size is not None and accepts_img:
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
                ap = _pos_embed_num_patches(inst)
                if ap is not None and ap != target_patches:
                    del inst
                    continue
            try:
                r = inst.load_state_dict(stripped_state, strict=False)
                score  = len(stripped_state) - len(r.missing_keys) - len(r.unexpected_keys)
                npar   = sum(p.numel() for p in inst.parameters())
            except Exception:
                score, npar = 0, 0
            if score > best_score or (score == best_score and npar > best_params):
                best_score, best_params = score, npar
                best_cls, best_instance = cls, inst
                if img_size is not None:
                    print(
                        f"  [CorridorKey] '{cls_name}' img_size={img_size} "
                        f"→ score {score}/{len(stripped_state)}, {npar/1e6:.1f} M"
                    )
            else:
                del inst
            break

    if best_cls is None or best_instance is None:
        raise RuntimeError(
            f"\nNo class matched {sorted(required_names)} in {module.__file__}.\n"
            f"Found: {[n for n, _ in candidates]}\n"
            "Run  uv run python nuke/inspect_model.py  for a diagnostic."
        )

    print(
        f"  [CorridorKey] Selected '{best_cls.__name__}'  "
        f"(score {best_score}/{len(stripped_state)}, {best_params/1e6:.1f} M params)"
    )
    return best_cls, best_instance


# ---------------------------------------------------------------------------
# Output format adapter
# ---------------------------------------------------------------------------

class _ModelAdapter(nn.Module):
    """Normalises model output to (alpha_coarse, fg_coarse, alpha_fine, fg_fine)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.inner(x)
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
            if len(vals) >= 2:
                return (vals[0], vals[1], vals[0], vals[1])
            raise ValueError(f"Unexpected dict keys: {list(out.keys())}")
        if isinstance(out, (list, tuple)):
            if len(out) == 4:
                return (out[0], out[1], out[2], out[3])
            if len(out) == 2:
                return (out[0], out[1], out[0], out[1])
        raise ValueError(f"Unexpected output type {type(out)}.")


# ---------------------------------------------------------------------------
# Stub — tests / CI only
# ---------------------------------------------------------------------------

class _StubInner(nn.Module):
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


# ---------------------------------------------------------------------------
# Main wrapper — all three bugs fixed
# ---------------------------------------------------------------------------

class CorridorKeyNukeWrapper(nn.Module):
    """
    TorchScript-traceable Nuke Cattery wrapper for CorridorKey v1.0.
    Compatible with Nuke 15.1+ and Nuke 17.0+.

    BUG 2 FIX: class-level type annotations (not just instance annotations in
    __init__).  TorchScript requires class-level annotations to expose typed
    attributes via prim::GetAttr.  Without them, Nuke 15 CatFileCreator
    reports "parameter not found" and the knobs are disconnected.
    """

    # ── BUG 2 FIX: class-level annotations ────────────────────────────────
    # These must be here at CLASS scope, not only in __init__.
    # This is what makes Nuke 15 CatFileCreator discover the attributes,
    # and what causes TorchScript to generate dynamic prim::GetAttr reads
    # rather than baking the values as CONSTANTS.
    despill_strength: float
    gamma_input:      int
    refiner_strength: float

    _MODEL_H: int = 2048
    _MODEL_W: int = 2048

    def __init__(
        self,
        checkpoint_path: str = "",
        despill_strength: float = 0.0,
        gamma_input: int = 0,
        refiner_strength: float = 1.0,
    ) -> None:
        super().__init__()

        # Assign without re-annotating (class-level annotation covers it)
        self.despill_strength = despill_strength
        self.gamma_input      = gamma_input
        self.refiner_strength = refiner_strength

        self.model = _ModelAdapter(self._load_model(checkpoint_path))

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(checkpoint_path: str) -> nn.Module:
        if not checkpoint_path:
            return _StubInner()

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"\nCheckpoint not found: {checkpoint_path}\n\n"
                "Download: uv run python nuke/download_checkpoint.py\n"
            )
        size_bytes = os.path.getsize(checkpoint_path)
        if size_bytes < 1_000_000:
            lfs_hint = ""
            try:
                with open(checkpoint_path, "rb") as f:
                    header = f.read(200).decode("utf-8", errors="replace")
                if "git-lfs" in header or "oid sha256" in header:
                    lfs_hint = "\nThis is a Git LFS pointer, not the real binary.\n"
            except Exception:
                pass
            raise ValueError(
                f"\nCheckpoint is {size_bytes:,} bytes — too small.\n"
                f"{lfs_hint}"
                "Download: uv run python nuke/download_checkpoint.py\n"
            )

        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CorridorKey")
        )
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        try:
            _mt_mod = importlib.import_module(
                "CorridorKeyModule.core.model_transformer"
            )
        except ImportError as exc:
            raise ImportError(
                f"\nCould not import CorridorKeyModule.core.model_transformer\n"
                f"Error: {exc}\n\nRun: uv sync\n"
            ) from exc

        try:
            raw = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"\ntorch.load failed: {exc}\n\n"
                "Re-download: uv run python nuke/download_checkpoint.py\n"
            ) from exc

        if isinstance(raw, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in raw:
                    raw = raw[key]
                    break

        raw = _strip_orig_mod(raw)

        try:
            _ModelClass, net = _discover_model_class(_mt_mod, raw)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

        try:
            net.load_state_dict(raw, strict=True)
        except RuntimeError as exc:
            state_tops = sorted({k.split(".")[0] for k in raw})
            model_tops = sorted(_top_level_submodule_names(net))
            raise RuntimeError(
                f"\nload_state_dict failed for '{_ModelClass.__name__}'.\n"
                f"  State dict tops: {state_tops}\n"
                f"  Model tops:      {model_tops}\n\n"
                f"Original error: {exc}\n\n"
                "Run  uv run python nuke/inspect_model.py  for a diagnostic.\n"
            ) from exc

        net.eval()
        n_params = sum(p.numel() for p in net.parameters())
        print(
            f"  [CorridorKey] Loaded {size_bytes/1e6:.0f} MB  "
            f"→ '{_ModelClass.__name__}'  ({n_params/1e6:.1f} M params)"
        )
        return net

    # ------------------------------------------------------------------
    # Color math helpers (TorchScript-safe)
    # ------------------------------------------------------------------

    def _linear_to_srgb(self, x: torch.Tensor) -> torch.Tensor:
        """IEC 61966-2-1: linear → sRGB."""
        x  = torch.clamp(x, 0.0, 1.0)
        lo = x * 12.92
        hi = 1.055 * torch.pow(torch.clamp(x, min=1e-9), 1.0 / 2.4) - 0.055
        return torch.clamp(torch.where(x <= 0.0031308, lo, hi), 0.0, 1.0)

    def _despill_green(self, fg: torch.Tensor) -> torch.Tensor:
        """
        Average green despill.
        BUG 3 FIX: no if-branch. Always executed. At despill_strength=0 the
        blend factor is 0.0 — mathematically identical to a bypass.
        trace() always records these ops → knob value read via prim::GetAttr
        at runtime (thanks to class-level annotation).
        """
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
    # forward() — Nuke 15 + 17 compatible
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : [1, 4, H, W]  ch0-2 = plate RGB,  ch3 = alpha hint
        out : [1, 4, H, W]  ch0-2 = sRGB FG,     ch3 = linear alpha

        Nuke 15 / 17 compatibility notes:

        Device (BUG 1 fix):
          All intermediate tensors derive their device from x (the input).
          No hardcoded .cuda() or .cpu() calls. The hiera CONSTANTS patch
          + CPU tracing ensures no device-specific constants are baked.

        gamma_input (BUG 3 fix — NO if-branch):
          ALWAYS compute both the raw plate and the sRGB-encoded plate.
          Blend by gamma_input read via prim::GetAttr at runtime:
            gamma_input=0 → plate * 1.0 + srgb * 0.0 = passthrough
            gamma_input=1 → plate * 0.0 + srgb * 1.0 = sRGB-encoded
          trace() records both computational paths. The knob value selects
          between them continuously without any baked branch.

        despill_strength (BUG 3 fix — NO if-branch):
          _despill_green() is ALWAYS called. At strength=0 the blend factor
          is 0.0 → identity. trace() records the despill ops → knob works.

        refiner_strength:
          Already uses arithmetic (s * delta). With class-level annotation,
          s is now a dynamic prim::GetAttr read instead of a baked CONSTANT.
        """
        dtype  = x.dtype
        H: int = x.shape[2]
        W: int = x.shape[3]

        plate = x[:, 0:3, :, :]
        hint  = x[:, 3:4, :, :]

        # ── BUG 3 FIX: gamma_input — arithmetic blend, no if-branch ────────
        # Both paths are ALWAYS recorded by trace().
        # gi is read via prim::GetAttr at runtime (class-level annotation).
        gi         = float(self.gamma_input)          # 0.0 or 1.0
        plate_srgb = self._linear_to_srgb(plate)     # sRGB-encoded version
        plate      = plate * (1.0 - gi) + plate_srgb * gi

        model_in = F.interpolate(
            torch.cat([plate, hint], dim=1),
            size=(self._MODEL_H, self._MODEL_W),
            mode="bilinear", align_corners=False,
        )

        alpha_c, fg_c, alpha_f, fg_f = self.model(model_in)

        s     = self.refiner_strength                 # dynamic prim::GetAttr
        alpha = torch.clamp(alpha_c + s * (alpha_f - alpha_c), 0.0, 1.0)
        fg    = torch.clamp(fg_c    + s * (fg_f    - fg_c),    0.0, 1.0)

        alpha = F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
        fg    = F.interpolate(fg,    size=(H, W), mode="bilinear", align_corners=False)

        # ── BUG 3 FIX: despill — always called, no if-branch ────────────────
        fg = self._despill_green(fg)

        return torch.cat([fg, alpha], dim=1).contiguous().to(dtype)
