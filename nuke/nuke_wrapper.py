###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/nikopueringer/CorridorKey
###############################################################################
"""
nuke_wrapper.py
===============
TorchScript-compatible Nuke Cattery wrapper for CorridorKey v1.0.

Drop into the nuke/ subfolder of nikopueringer/CorridorKey.
Upstream files are never modified.

Nuke .cat I/O
-------------
  Input  [1, 4, H, W]  rgba.red/.green/.blue = plate,  rgba.alpha = hint
  Output [1, 4, H, W]  rgba.red/.green/.blue = sRGB FG, rgba.alpha = linear alpha

Custom knobs (attribute names must match CatFileCreator "Name" exactly)
  despill_strength  float  0-10   green despill intensity
  gamma_input       int    0/1    0 = sRGB plate,  1 = linear EXR plate
  refiner_strength  float  0-1    0 = coarse only, 1 = full CNN refinement

Architecture (from checkpoint key analysis)
-------------------------------------------
  Outer class  : GreenFormer(img_size=2048, ...)
  Sub-modules  : encoder (ViT, 24 blocks), alpha_decoder (SegFormer MLP),
                 fg_decoder (SegFormer MLP), refiner (CNN)
  Checkpoint   : saved from torch.compile() → '_orig_mod.' key prefix
  img_size bug : patch_embed uses kernel=7, stride=4  (not stride==kernel)
                 img_size must be inferred from the *stride*, not the kernel.
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
# Step 1 — strip torch.compile() prefix
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
# Step 2 — discover the model class by sub-module name matching + probe score
# ---------------------------------------------------------------------------

def _top_level_submodule_names(net: nn.Module) -> set:
    return {name for name, _ in net.named_children()}


def _read_patch_stride(instance: nn.Module) -> Optional[int]:
    """
    Read the actual stride of the patch-embedding conv from a live model instance.

    Traverses: instance.encoder.model.patch_embed.proj  (standard ViT path)
    Falls back to a deep search for any Conv2d with 'patch' in its path.

    Returns the stride as an int, or None if not found.
    """
    # Primary path: standard ViT layout used by GreenFormer
    try:
        conv = instance.encoder.model.patch_embed.proj
        if isinstance(conv, nn.Conv2d):
            s = conv.stride
            return s[0] if isinstance(s, (tuple, list)) else int(s)
    except AttributeError:
        pass

    # Fallback: search the whole module tree
    for name, mod in instance.named_modules():
        if isinstance(mod, nn.Conv2d) and "patch" in name.lower():
            s = mod.stride
            return s[0] if isinstance(s, (tuple, list)) else int(s)

    return None


def _pos_embed_num_patches(instance: nn.Module) -> Optional[int]:
    """Return num_patches from this instance's pos_embed, or None."""
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
    Find the nn.Module subclass in model_transformer.py that matches the
    checkpoint, instantiated with the correct constructor arguments.

    img_size fix
    ------------
    The patch embedding uses Conv2d(kernel=7, stride=4).  Inferring img_size
    from kernel_size gives the wrong value (3584 instead of 2048).

    Correct approach:
      1. Instantiate the candidate class with NO args (default img_size).
      2. Read the actual patch stride from instance.encoder.model.patch_embed.proj.stride
      3. Compute: img_size = sqrt(target_num_patches) * patch_stride
      4. Re-instantiate with the correct img_size.
      5. Run load_state_dict(strict=False) as a probe to confirm.

    If the class does not accept img_size, or stride cannot be read,
    we fall back to trying a list of common img_size values.
    """
    required_names = {k.split(".")[0] for k in stripped_state}
    print(
        f"  [CorridorKey] Required sub-modules from checkpoint: "
        f"{sorted(required_names)}"
    )

    # Target num_patches from checkpoint pos_embed
    target_patches: Optional[int] = None
    for k, v in stripped_state.items():
        if "pos_embed" in k:
            target_patches = v.shape[1]
            break

    # Gather public nn.Module subclasses defined in this module
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
            f"No public nn.Module subclasses found in {module.__file__}.\n"
            f"Names: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    best_cls:      Optional[type]      = None
    best_instance: Optional[nn.Module] = None
    best_score     = -1
    best_params    = -1

    for cls_name, cls in candidates:
        # Does this class accept constructor arguments?
        try:
            sig = inspect.signature(cls.__init__)
            accepted_params = {
                n: p for n, p in sig.parameters.items()
                if n != "self"
            }
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in accepted_params.values()
            )
            required_args = [
                n for n, p in accepted_params.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
        except (ValueError, TypeError):
            accepted_params = {}
            accepts_kwargs  = False
            required_args   = []

        if required_args:
            continue  # Cannot auto-instantiate

        # ── Build the list of img_size values to try ───────────────────
        img_size_candidates: List[Optional[int]] = [None]  # None = no-arg default

        if target_patches is not None:
            target_grid = math.isqrt(target_patches)

            if ("img_size" in accepted_params) or accepts_kwargs:
                # Strategy A: instantiate with default, read stride, compute img_size
                try:
                    default_inst = cls()
                    patch_stride = _read_patch_stride(default_inst)
                    if patch_stride is not None and patch_stride > 0:
                        computed_img_size = target_grid * patch_stride
                        img_size_candidates.insert(0, computed_img_size)
                        print(
                            f"  [CorridorKey] Inferred img_size={computed_img_size} "
                            f"from patch stride={patch_stride} "
                            f"(num_patches={target_patches}={target_grid}²)"
                        )
                    del default_inst
                except Exception:
                    pass

                # Strategy B: common multiples of target_grid as fallback
                for stride_guess in (4, 2, 8, 1, 16, 3, 7, 14, 32):
                    gs = target_grid * stride_guess
                    if gs not in img_size_candidates:
                        img_size_candidates.append(gs)

        # ── Try each img_size ──────────────────────────────────────────
        instance: Optional[nn.Module] = None
        for img_size in img_size_candidates:
            kwargs: Dict[str, Any] = {}
            if img_size is not None:
                if "img_size" in accepted_params or accepts_kwargs:
                    kwargs["img_size"] = img_size

            try:
                inst = cls(**kwargs)
                inst.eval()
            except Exception:
                continue

            # Must have ALL required sub-modules
            if not required_names.issubset(_top_level_submodule_names(inst)):
                del inst
                continue

            # Check pos_embed matches — if not, skip this img_size
            if target_patches is not None:
                actual_patches = _pos_embed_num_patches(inst)
                if actual_patches is not None and actual_patches != target_patches:
                    del inst
                    continue

            # This img_size gives the right pos_embed shape → probe the full load
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
                if img_size is not None:
                    print(
                        f"  [CorridorKey] '{cls_name}' with img_size={img_size} "
                        f"→ score {score}/{len(stripped_state)}, "
                        f"{n_params/1e6:.1f} M params"
                    )
            else:
                del inst

            break  # Found a valid img_size for this class — stop trying others

    if best_cls is None or best_instance is None:
        raise RuntimeError(
            f"\nNo class in {module.__file__} matched all required "
            f"sub-modules {sorted(required_names)}.\n\n"
            f"Classes found: {[n for n, _ in candidates]}\n\n"
            "Run  uv run python nuke/inspect_model.py  for a full diagnostic."
        )

    print(
        f"  [CorridorKey] Selected '{best_cls.__name__}'  "
        f"(score {best_score}/{len(stripped_state)}, "
        f"{best_params/1e6:.1f} M params)"
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
            raise ValueError(f"Dict keys: {list(out.keys())}")

        if isinstance(out, (list, tuple)):
            if len(out) == 4:
                return (out[0], out[1], out[2], out[3])
            if len(out) == 2:
                return (out[0], out[1], out[0], out[1])

        raise ValueError(f"Unexpected output type {type(out)}.")


# ---------------------------------------------------------------------------
# Stub — for tests / CI when no checkpoint is supplied
# ---------------------------------------------------------------------------

class _StubInner(nn.Module):
    """Correct shape/range, meaningless values. Tests only."""

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
# Main wrapper
# ---------------------------------------------------------------------------

class CorridorKeyNukeWrapper(nn.Module):
    """TorchScript-traceable Nuke Cattery wrapper for CorridorKey v1.0."""

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

        # Knob attributes — names MUST match CatFileCreator "Name" fields exactly
        self.despill_strength: float = despill_strength
        self.gamma_input: int        = gamma_input
        self.refiner_strength: float = refiner_strength

        self.model = _ModelAdapter(self._load_model(checkpoint_path))

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(checkpoint_path: str) -> nn.Module:
        """
        Load the full CorridorKey model + weights.

        Three non-obvious problems solved here:
          1. _orig_mod. prefix from torch.compile() — strip before load
          2. Model class name unknown — discover by sub-module matching
          3. img_size defaults wrong — read patch stride from a live instance,
             compute correct img_size, re-instantiate before loading weights
        """
        if not checkpoint_path:
            return _StubInner()

        # ── Validate file ─────────────────────────────────────────────
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
                f"\nCheckpoint is {size_bytes:,} bytes — too small (expected ~300 MB).\n"
                f"{lfs_hint}"
                "Download: uv run python nuke/download_checkpoint.py\n"
            )

        # ── Ensure repo root is importable ────────────────────────────
        _repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        # ── Import model_transformer ──────────────────────────────────
        try:
            _mt_mod = importlib.import_module(
                "CorridorKeyModule.core.model_transformer"
            )
        except ImportError as exc:
            raise ImportError(
                f"\nCould not import CorridorKeyModule.core.model_transformer\n"
                f"Error: {exc}\n\n"
                "Run from the repo root:  uv sync\n"
                f"Repo root tried: {_repo_root}\n"
            ) from exc

        # ── Load raw checkpoint ───────────────────────────────────────
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

        # ── Strip _orig_mod. prefix ────────────────────────────────────
        raw = _strip_orig_mod(raw)

        # ── Discover class and correctly-instantiated model ────────────
        #
        # _discover_model_class:
        #   1. Reads patch stride from a default instance (not from weight shape)
        #   2. Computes img_size = sqrt(num_patches) * patch_stride
        #   3. Re-instantiates with the correct img_size
        #   4. Verifies pos_embed shape matches before probing load
        try:
            _ModelClass, net = _discover_model_class(_mt_mod, raw)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

        # ── Load weights (strict) ────────────────────────────────────
        try:
            net.load_state_dict(raw, strict=True)
        except RuntimeError as exc:
            state_tops = sorted({k.split(".")[0] for k in raw})
            model_tops = sorted(_top_level_submodule_names(net))
            raise RuntimeError(
                f"\nload_state_dict(strict=True) failed for '{_ModelClass.__name__}'.\n"
                f"  State dict top-level keys : {state_tops}\n"
                f"  Model  top-level children : {model_tops}\n\n"
                f"Original error: {exc}\n\n"
                "Run  uv run python nuke/inspect_model.py  for a full diagnostic.\n"
            ) from exc

        net.eval()
        n_params = sum(p.numel() for p in net.parameters())
        print(
            f"  [CorridorKey] Loaded {size_bytes/1e6:.0f} MB checkpoint  "
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
        """Average green despill — only G channel is modified."""
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
    # forward()
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : [1, 4, H, W]  ch0-2 = plate RGB,  ch3 = alpha hint
        out : [1, 4, H, W]  ch0-2 = sRGB FG,     ch3 = linear alpha
        """
        dtype  = x.dtype
        H: int = x.shape[2]
        W: int = x.shape[3]

        plate = x[:, 0:3, :, :]
        hint  = x[:, 3:4, :, :]

        if self.gamma_input == 1:
            plate = self._linear_to_srgb(plate)

        model_in = F.interpolate(
            torch.cat([plate, hint], dim=1),
            size=(self._MODEL_H, self._MODEL_W),
            mode="bilinear", align_corners=False,
        )

        alpha_c, fg_c, alpha_f, fg_f = self.model(model_in)

        s     = self.refiner_strength
        alpha = torch.clamp(alpha_c + s * (alpha_f - alpha_c), 0.0, 1.0)
        fg    = torch.clamp(fg_c    + s * (fg_f    - fg_c),    0.0, 1.0)

        alpha = F.interpolate(alpha, size=(H, W), mode="bilinear", align_corners=False)
        fg    = F.interpolate(fg,    size=(H, W), mode="bilinear", align_corners=False)

        if self.despill_strength > 0.0:
            fg = self._despill_green(fg)

        return torch.cat([fg, alpha], dim=1).contiguous().to(dtype)
    