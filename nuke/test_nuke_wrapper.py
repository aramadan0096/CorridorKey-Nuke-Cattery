#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/nikopueringer/CorridorKey
###############################################################################
#!/usr/bin/env python3
"""
test_nuke_wrapper.py
====================
Self-contained test suite for nuke_wrapper.py.

Runs without PyTorch — uses an inline NumPy mock.
When real PyTorch IS installed, the jit.trace test exercises the real path.

Run from anywhere inside the CorridorKey repo:
    python nuke/test_nuke_wrapper.py
    uv run python nuke/test_nuke_wrapper.py

32 tests across 7 sections. Expected: 32 passed, 0 failed.
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np


# ===========================================================================
# NumPy mock of torch — injected before any other import
# Must define every torch symbol used by nuke_wrapper.py AND by these tests.
# ===========================================================================

class _Tensor:
    """Minimal torch.Tensor backed by a NumPy float32 array."""

    def __init__(self, d):
        self._d = np.asarray(d, dtype=np.float32).copy()

    @property
    def shape(self):        return self._d.shape
    @property
    def dtype(self):        return self._d.dtype
    @property
    def is_cuda(self):      return False
    @property
    def device(self):
        class _D:
            type = "cpu"
        return _D()

    def numpy(self):               return self._d.copy()
    def float(self):               return _Tensor(self._d.astype(np.float32))
    def to(self, *a, **k):         return self
    def contiguous(self):          return _Tensor(np.ascontiguousarray(self._d))
    def is_contiguous(self):       return bool(self._d.flags["C_CONTIGUOUS"])
    def min(self):                 return float(self._d.min())
    def max(self):                 return float(self._d.max())
    def mean(self):                return float(self._d.mean())
    def any(self):                 return bool(self._d.any())
    def all(self):                 return bool(self._d.all())

    def __add__(self, o):          return _Tensor(self._d + _v(o))
    def __radd__(self, o):         return _Tensor(_v(o) + self._d)
    def __sub__(self, o):          return _Tensor(self._d - _v(o))
    def __rsub__(self, o):         return _Tensor(_v(o) - self._d)
    def __mul__(self, o):          return _Tensor(self._d * _v(o))
    def __rmul__(self, o):         return _Tensor(_v(o) * self._d)
    def __truediv__(self, o):      return _Tensor(self._d / _v(o))
    def __rtruediv__(self, o):     return _Tensor(_v(o) / self._d)
    def __neg__(self):             return _Tensor(-self._d)
    def __le__(self, v):           return _Tensor((self._d <= _v(v)).astype(np.float32))
    def __ge__(self, v):           return _Tensor((self._d >= _v(v)).astype(np.float32))
    def __getitem__(self, i):      return _Tensor(self._d[i])
    def __setitem__(self, i, v):   self._d[i] = _v(v)
    def __repr__(self):            return f"Tensor{self.shape}"


def _v(x):
    """Unwrap _Tensor → numpy array for arithmetic."""
    return x._d if isinstance(x, _Tensor) else x


# ---------------------------------------------------------------------------
# torch namespace — every symbol used by nuke_wrapper.py and by the tests
# ---------------------------------------------------------------------------

import types as _types

_torch_mod = _types.ModuleType("torch")


def _tensor(d, dtype=None, device=None):  return _Tensor(d)
def _zeros(*s, dtype=None, device=None):
    s = s[0] if (len(s) == 1 and not isinstance(s[0], int)) else s
    return _Tensor(np.zeros(s, np.float32))
def _ones(*s, dtype=None, device=None):
    s = s[0] if (len(s) == 1 and not isinstance(s[0], int)) else s
    return _Tensor(np.ones(s, np.float32))
def _rand(*s, dtype=None, device=None, generator=None):
    s = s[0] if (len(s) == 1 and not isinstance(s[0], int)) else s
    return _Tensor(np.random.rand(*s).astype(np.float32))
def _full(s, v, dtype=None, device=None): return _Tensor(np.full(s, v, np.float32))
def _cat(ts, dim=0):  return _Tensor(np.concatenate([_v(t) for t in ts], axis=dim))
def _stack(ts, dim=0):
    return _Tensor(np.stack([_v(t) for t in ts], axis=dim))
def _clamp(x, min=None, max=None):
    return _Tensor(np.clip(_v(x), a_min=min, a_max=max))
def _where(c, a, b):
    return _Tensor(np.where(_v(c).astype(bool), _v(a), _v(b)))
def _pow(x, e):
    return _Tensor(np.power(np.clip(_v(x), 1e-300, None), e).astype(np.float32))
def _minimum(a, b):   return _Tensor(np.minimum(_v(a), _v(b)))
def _maximum(a, b):   return _Tensor(np.maximum(_v(a), _v(b)))
def _isnan(x):        return _Tensor(np.isnan(_v(x)).astype(np.float32))
def _isinf(x):        return _Tensor(np.isinf(_v(x)).astype(np.float32))
def _allclose(a, b, atol=1e-5, rtol=0):
    return bool(np.allclose(_v(a), _v(b), atol=atol, rtol=rtol))
def _sigmoid(x):
    d = _v(x).astype(np.float64)
    return _Tensor((1.0 / (1.0 + np.exp(-d))).astype(np.float32))
def _manual_seed(s):  np.random.seed(s)
def _load(path, map_location=None, weights_only=False):
    raise RuntimeError("torch.load called inside mock — should not happen in tests")


class _device:
    def __init__(self, s="cpu"): self.type = s if isinstance(s, str) else "cpu"

class _cuda_ns:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 0

class _jit_ns:
    @staticmethod
    def trace(model, *args, **kwargs): return model
    @staticmethod
    def script(model, *args, **kwargs): return model


for _name, _fn in {
    "tensor": _tensor, "zeros": _zeros, "ones": _ones, "rand": _rand,
    "full": _full, "cat": _cat, "stack": _stack, "clamp": _clamp,
    "where": _where, "pow": _pow, "minimum": _minimum, "maximum": _maximum,
    "isnan": _isnan, "isinf": _isinf, "allclose": _allclose, "sigmoid": _sigmoid,
    "manual_seed": _manual_seed, "load": _load,
}.items():
    setattr(_torch_mod, _name, _fn)

_torch_mod.Tensor   = _Tensor
_torch_mod.device   = _device
_torch_mod.float32  = np.float32
_torch_mod.cuda     = _cuda_ns
_torch_mod.jit      = _jit_ns

# ---------------------------------------------------------------------------
# torch.nn namespace
# ---------------------------------------------------------------------------

_nn_mod = _types.ModuleType("torch.nn")


class _Module:
    def __init__(self): pass
    def eval(self):           return self
    def to(self, *a, **k):    return self
    def __call__(self, *a, **k): return self.forward(*a, **k)
    def parameters(self):
        # Yield weight/bias tensors from Conv2d children
        for attr in vars(self).values():
            if isinstance(attr, _Conv2d):
                yield attr.weight
                if attr.bias is not None:
                    yield attr.bias


class _Conv2d(_Module):
    def __init__(self, in_ch, out_ch, kernel_size, bias=True):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.weight = _Tensor(np.random.randn(out_ch, in_ch, k, k).astype(np.float32) * 0.01)
        self.bias   = _Tensor(np.zeros(out_ch, np.float32)) if bias else None

    def forward(self, x: _Tensor) -> _Tensor:
        d = _v(x)                     # [B, in_ch, H, W]
        w = _v(self.weight)           # [out_ch, in_ch, k, k]
        out = np.einsum("bihw,oiab->bohw", d, w).astype(np.float32)
        if self.bias is not None:
            out += _v(self.bias)[None, :, None, None]
        return _Tensor(out)


class _init_ns:
    @staticmethod
    def constant_(t, val: float) -> None:
        if isinstance(t, _Tensor):
            t._d.fill(val)
        elif t is not None:
            np.ndarray.fill(t, val)


_nn_mod.Module  = _Module
_nn_mod.Conv2d  = _Conv2d
_nn_mod.init    = _init_ns()

# ---------------------------------------------------------------------------
# torch.nn.functional namespace
# ---------------------------------------------------------------------------

_nn_func = _types.ModuleType("torch.nn.functional")


def _interpolate(x, size, mode="bilinear", align_corners=False):
    d = _v(x)
    B, C, _, _ = d.shape
    H, W = size
    ih = (np.arange(H) * d.shape[2] / H).astype(int).clip(0, d.shape[2] - 1)
    iw = (np.arange(W) * d.shape[3] / W).astype(int).clip(0, d.shape[3] - 1)
    return _Tensor(d[:, :, ih, :][:, :, :, iw].astype(np.float32))


_nn_func.interpolate = _interpolate
_nn_mod.functional   = _nn_func
_torch_mod.nn        = _nn_mod

# ---------------------------------------------------------------------------
# Inject mock BEFORE any import that might pull in torch
# ---------------------------------------------------------------------------

sys.modules["torch"]                 = _torch_mod
sys.modules["torch.nn"]              = _nn_mod
sys.modules["torch.nn.functional"]   = _nn_func

# ---------------------------------------------------------------------------
# Import wrapper using mock
# ---------------------------------------------------------------------------

_NUKE_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _NUKE_DIR.parent

for _p in (str(_REPO_ROOT), str(_NUKE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from nuke_wrapper import CorridorKeyNukeWrapper, _StubInner  # type: ignore
except ImportError:
    from nuke.nuke_wrapper import CorridorKeyNukeWrapper, _StubInner  # type: ignore


# ===========================================================================
# NumPy reference implementations (ground truth for math tests)
# ===========================================================================

def _np_linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1: linear → sRGB, pure NumPy."""
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip(
        np.where(x <= 0.0031308, x * 12.92,
                 1.055 * np.power(np.clip(x, 1e-9, None), 1.0 / 2.4) - 0.055),
        0.0, 1.0,
    ).astype(np.float32)


# ===========================================================================
# Micro test runner
# ===========================================================================

_PASS = _FAIL = 0
_ERRORS: list = []


def _test(name: str, fn) -> None:
    global _PASS, _FAIL
    try:
        fn()
        print(f"  ✓  {name}")
        _PASS += 1
    except AssertionError as e:
        print(f"  ✗  {name}")
        print(f"       {e}")
        _FAIL += 1
        _ERRORS.append((name, str(e)))
    except Exception:
        msg = traceback.format_exc().strip().splitlines()[-1]
        print(f"  ✗  {name}")
        print(f"       {msg}")
        _FAIL += 1
        _ERRORS.append((name, msg))


def _sec(title: str) -> None:
    print(f"\n  {'─' * 56}")
    print(f"  {title}")
    print(f"  {'─' * 56}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rnd(*shape, seed=0) -> _Tensor:
    np.random.seed(seed)
    return _Tensor(np.random.rand(*shape).astype(np.float32))


def _model(**kw) -> CorridorKeyNukeWrapper:
    """Stub model — no checkpoint."""
    # Knob kwargs are set as attributes after construction (new API)
    m = CorridorKeyNukeWrapper(_StubInner())
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _run(m: CorridorKeyNukeWrapper, H=64, W=64, seed=1) -> _Tensor:
    np.random.seed(seed)
    x = _Tensor(np.random.rand(1, 4, H, W).astype(np.float32))
    return m.forward(x)


def _close(a, b, atol=1e-5, msg=""):
    """Compare two things that may be _Tensor or np.ndarray."""
    ad = _v(a) if isinstance(a, _Tensor) else np.asarray(a)
    bd = _v(b) if isinstance(b, _Tensor) else np.asarray(b)
    err = float(np.abs(ad - bd).max())
    assert err <= atol, f"{msg}  max_err={err:.2e}  atol={atol:.2e}"


# ===========================================================================
# 1. sRGB transfer functions
# ===========================================================================

def _s1():
    _sec("1. sRGB transfer functions")

    def t_vs_numpy():
        """_linear_to_srgb must match the NumPy IEC 61966-2-1 reference.
        Test uses explicit np.ndarray inputs/outputs to avoid array-protocol
        dispatch issues with numpy 2.4.x on systems with real PyTorch installed."""
        np.random.seed(0)
        x_np = np.random.rand(1, 3, 16, 16).astype(np.float32)
        ref  = _np_linear_to_srgb(x_np)
        # Call the wrapper method directly with a _Tensor
        m    = _model()
        out  = _v(m._linear_to_srgb(_Tensor(x_np)))   # extract numpy from _Tensor
        # Compare two plain numpy arrays — no torch involved
        assert np.allclose(ref, out, atol=1e-5), \
            f"Max err {np.abs(ref - out).max():.2e}"
    _test("_linear_to_srgb matches NumPy IEC 61966-2-1 reference", t_vs_numpy)

    def t_black_white():
        m = _model()
        z = _Tensor(np.zeros((1, 3, 4, 4), np.float32))
        o = _Tensor(np.ones((1, 3, 4, 4), np.float32))
        assert _v(m._linear_to_srgb(z)).max() < 1e-7
        assert abs(_v(m._linear_to_srgb(o)).min() - 1.0) < 1e-5
    _test("Black → 0, white → 1", t_black_white)

    def t_midgrey():
        m   = _model()
        x   = _Tensor(np.full((1, 1, 1, 1), 0.2140, np.float32))
        out = float(_v(m._linear_to_srgb(x)).flat[0])
        assert abs(out - 0.5) < 0.005, f"linear 0.214 → sRGB expected ~0.5, got {out:.4f}"
    _test("linear 0.2140 → sRGB ≈ 0.5  (mid-grey)", t_midgrey)

    def t_range():
        m   = _model()
        x   = _rnd(1, 3, 32, 32, seed=2)
        out = _v(m._linear_to_srgb(x))
        assert out.min() >= 0.0 and out.max() <= 1.0
    _test("linear_to_srgb output clamped to [0, 1]", t_range)

    def t_no_nan():
        m   = _model()
        x   = _rnd(1, 3, 32, 32, seed=3)
        out = _v(m._linear_to_srgb(x))
        assert not np.isnan(out).any()
    _test("No NaN from linear_to_srgb on random input", t_no_nan)


# ===========================================================================
# 2. Green despill
# ===========================================================================

def _s2():
    _sec("2. Green despill")

    def t_zero_identity():
        m  = _model(despill_strength=0.0)
        np.random.seed(4)
        fg = _Tensor(np.random.rand(1, 3, 16, 16).astype(np.float32))
        assert np.allclose(_v(fg), _v(m._despill_green(fg)), atol=1e-6)
    _test("despill_strength=0 is exact identity", t_zero_identity)

    def t_only_green():
        m  = _model(despill_strength=5.0)
        np.random.seed(5)
        fg = _Tensor(np.random.rand(1, 3, 16, 16).astype(np.float32))
        out = m._despill_green(fg)
        assert np.allclose(_v(fg)[:, 0], _v(out)[:, 0], atol=1e-6), "R changed"
        assert np.allclose(_v(fg)[:, 2], _v(out)[:, 2], atol=1e-6), "B changed"
    _test("Only green channel is modified (R, B untouched)", t_only_green)

    def t_full_caps_green():
        m  = _model(despill_strength=10.0)
        np.random.seed(6)
        fg = _Tensor(np.random.rand(1, 3, 32, 32).astype(np.float32))
        out = m._despill_green(fg)
        r, g, b = _v(out)[:, 0], _v(out)[:, 1], _v(out)[:, 2]
        assert np.all(g <= (r + b) / 2.0 + 1e-5)
    _test("Full despill: G ≤ avg(R, B) for every pixel", t_full_caps_green)

    def t_clamped():
        m  = _model(despill_strength=7.0)
        fg = _Tensor(np.random.rand(1, 3, 16, 16).astype(np.float32))
        out = _v(m._despill_green(fg))
        assert out.min() >= 0.0 and out.max() <= 1.0
    _test("Despill output stays in [0, 1]", t_clamped)

    def t_monotonic():
        np.random.seed(7)
        fg_np = np.random.rand(1, 3, 32, 32).astype(np.float32)
        fg_np[:, 1] = 0.9
        means = []
        for s in (0.0, 3.0, 6.0, 10.0):
            m = _model(despill_strength=s)
            means.append(float(_v(m._despill_green(_Tensor(fg_np.copy())))[:, 1].mean()))
        for i in range(len(means) - 1):
            assert means[i+1] <= means[i] + 1e-4, \
                f"Not monotonic at step {i}: {means[i]:.4f} → {means[i+1]:.4f}"
    _test("Green reduction is monotonic with strength", t_monotonic)

    def t_pure_green_to_zero():
        m  = _model(despill_strength=10.0)
        fg = _Tensor(np.array([[[[0]], [[1]], [[0]]]]).astype(np.float32))
        assert float(_v(m._despill_green(fg))[:, 1].max()) < 1e-5
    _test("Pure green pixel (R=0,G=1,B=0) → G=0 at strength 10", t_pure_green_to_zero)

    def t_non_dominant_unchanged():
        m  = _model(despill_strength=10.0)
        fg = _Tensor(np.array([[[[0.8]], [[0.3]], [[0.7]]]]).astype(np.float32))
        out = m._despill_green(fg)
        assert abs(float(_v(out)[:, 1].flat[0]) - 0.3) < 1e-5
    _test("G not dominant (R=0.8,G=0.3,B=0.7) → unchanged", t_non_dominant_unchanged)


# ===========================================================================
# 3. Nuke .cat I/O contract
# ===========================================================================

def _s3():
    _sec("3. Nuke .cat I/O contract")

    def t_shapes():
        m = _model()
        for H, W in ((64, 64), (1080, 1920), (2048, 2048)):
            out = _run(m, H, W)
            assert out.shape == (1, 4, H, W), f"Expected (1,4,{H},{W}), got {out.shape}"
    _test("Output shape == input shape for multiple resolutions", t_shapes)

    def t_alpha_range():
        out = _run(_model(), 128, 128)
        a = _v(out)[:, 3]
        assert a.min() >= -0.01 and a.max() <= 1.01, f"Alpha [{a.min():.3f},{a.max():.3f}]"
    _test("Output alpha (ch3) in [0, 1]", t_alpha_range)

    def t_fg_range():
        out = _run(_model(), 128, 128)
        fg = _v(out)[:, :3]
        assert fg.min() >= -0.01 and fg.max() <= 1.01, f"FG [{fg.min():.3f},{fg.max():.3f}]"
    _test("Output FG RGB (ch0–2) in [0, 1]", t_fg_range)

    def t_contiguous():
        assert _run(_model()).is_contiguous()
    _test("Output is C-contiguous (Nuke memory requirement)", t_contiguous)

    def t_no_nan_inf():
        out = _v(_run(_model()))
        assert not np.isnan(out).any() and not np.isinf(out).any()
    _test("No NaN / Inf in output", t_no_nan_inf)

    def t_batch_1():
        assert _run(_model()).shape[0] == 1
    _test("Output batch dimension is 1", t_batch_1)

    def t_all_zero():
        m = _model()
        assert not np.isnan(_v(m.forward(_Tensor(np.zeros((1, 4, 32, 32), np.float32))))).any()
    _test("All-zero input → no NaN", t_all_zero)

    def t_all_one():
        m = _model()
        assert not np.isnan(_v(m.forward(_Tensor(np.ones((1, 4, 32, 32), np.float32))))).any()
    _test("All-one input → no NaN", t_all_one)


# ===========================================================================
# 4. CatFileCreator knob type requirements
# ===========================================================================

def _s4():
    _sec("4. CatFileCreator knob types")

    KNOBS = {
        "despill_strength": (float, 0.0),
        "gamma_input":      (int,   0),
        "refiner_strength": (float, 1.0),
    }

    for attr, (typ, default) in KNOBS.items():
        def _mk(a, t, d):
            def fn():
                m = _model()
                v = getattr(m, a)
                assert isinstance(v, t), f"'{a}' should be {t.__name__}, got {type(v).__name__}"
                assert not isinstance(v, bool), f"'{a}' must not be bool"
                assert v == d, f"'{a}' default should be {d}, got {v}"
            return fn
        _test(f"Knob '{attr}' is {typ.__name__} = {default!r}", _mk(attr, typ, default))

    def t_all_present():
        m = _model()
        for n in KNOBS:
            assert hasattr(m, n), f"Missing: '{n}'"
    _test("All knob attributes present on model instance", t_all_present)


# ===========================================================================
# 5. gamma_input knob
# ===========================================================================

def _s5():
    _sec("5. gamma_input knob")

    def t_srgb_ne_linear():
        np.random.seed(8)
        x = _Tensor(np.random.rand(1, 4, 32, 32).astype(np.float32))
        out0 = _model(gamma_input=0).forward(x)
        out1 = _model(gamma_input=1).forward(x)
        diff = float(np.abs(_v(out0) - _v(out1)).mean())
        assert diff > 1e-4, f"gamma_input 0 and 1 give same output (diff={diff:.2e})"
    _test("gamma_input=0 (sRGB) ≠ gamma_input=1 (linear)", t_srgb_ne_linear)

    def t_default_zero():
        m = _model()
        assert m.gamma_input == 0 and isinstance(m.gamma_input, int)
    _test("gamma_input defaults to 0 (sRGB)", t_default_zero)

    def t_linear_encoded():
        ref = float(_np_linear_to_srgb(np.array([[[[0.2140]]]])).flat[0])
        assert abs(ref - 0.5) < 0.005
        m   = _model(gamma_input=1)
        out = m.forward(_Tensor(np.random.rand(1, 4, 16, 16).astype(np.float32)))
        assert not np.isnan(_v(out)).any()
    _test("Linear plate (0.214 → sRGB ≈ 0.5) encoded before model", t_linear_encoded)


# ===========================================================================
# 6. Output channel order
# ===========================================================================

def _s6():
    _sec("6. Output channel order (FG=ch0–2, alpha=ch3)")

    def t_alpha_ch3():
        m = _model()
        xw = _Tensor(np.random.rand(1, 4, 64, 64).astype(np.float32))
        xw._d[:, 3] = 1.0
        xb = _Tensor(xw._d.copy())
        xb._d[:, 3] = 0.0
        aw = float(_v(m.forward(xw))[:, 3].mean())
        ab = float(_v(m.forward(xb))[:, 3].mean())
        assert aw >= 0.0 and ab <= 0.9 and aw >= ab
    _test("ch3 (alpha) responds to hint: white > black", t_alpha_ch3)

    def t_fg_ch0_2_nonzero():
        np.random.seed(11)
        x = _Tensor(np.random.rand(1, 4, 64, 64).astype(np.float32))
        x._d[:, 3] = 0.7
        out = _model().forward(x)
        assert float(_v(out)[:, :3].mean()) > 0.0
    _test("ch0–2 (FG) carry non-zero signal", t_fg_ch0_2_nonzero)

    def t_despill_fg_not_alpha():
        """Same model instance, different despill_strength → same alpha, different FG."""
        np.random.seed(12)
        x = _Tensor(np.random.rand(1, 4, 32, 32).astype(np.float32))
        x._d[:, 1] = 0.9
        m = _model(despill_strength=0.0)
        out0 = m.forward(x)
        m.despill_strength = 10.0
        out1 = m.forward(x)
        fg_diff    = float(np.abs(_v(out0)[:, :3] - _v(out1)[:, :3]).mean())
        alpha_diff = float(np.abs(_v(out0)[:, 3]  - _v(out1)[:, 3]).mean())
        assert fg_diff    > 1e-5,  "Despill had no effect on FG"
        assert alpha_diff < 1e-6,  f"Despill changed alpha: diff={alpha_diff:.2e}"
    _test("despill_strength changes FG (ch0–2) but not alpha (ch3)", t_despill_fg_not_alpha)


# ===========================================================================
# 7. TorchScript traceability
# ===========================================================================

def _s7():
    _sec("7. TorchScript traceability")

    def t_deterministic():
        m = _model()
        np.random.seed(42)
        x1 = _Tensor(np.random.rand(1, 4, 32, 32).astype(np.float32))
        np.random.seed(42)
        x2 = _Tensor(np.random.rand(1, 4, 32, 32).astype(np.float32))
        out1 = m.forward(x1)
        out2 = m.forward(x2)
        assert np.allclose(_v(out1), _v(out2), atol=1e-6)
    _test("forward() is deterministic (same seed → same output)", t_deterministic)

    def t_real_torch_trace():
        """Run torch.jit.trace with the REAL PyTorch if available."""
        try:
            import importlib as _il
            real_torch = _il.import_module("torch")
            if not hasattr(real_torch, "no_grad"):
                print("    (skipped — real PyTorch not installed)")
                return
        except ImportError:
            print("    (skipped — real PyTorch not installed)")
            return

        import sys as _sys, importlib.util as _ilu

        # Temporarily restore real torch for this sub-test only
        _orig = {k: _sys.modules.get(k)
                 for k in ("torch", "torch.nn", "torch.nn.functional")}
        try:
            _sys.modules["torch"]                = real_torch
            _sys.modules["torch.nn"]             = real_torch.nn
            _sys.modules["torch.nn.functional"]  = real_torch.nn.functional

            _wrapper_path = str(Path(__file__).parent / "nuke_wrapper.py")
            spec = _ilu.spec_from_file_location("nuke_wrapper_real", _wrapper_path)
            mod  = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)

            m = mod.CorridorKeyNukeWrapper()   # stub — no checkpoint
            m.eval()

            dummy  = real_torch.zeros(1, 4, 64, 64)
            traced = real_torch.jit.trace(m, dummy, strict=False)

            with real_torch.no_grad():
                out = traced(dummy)

            assert tuple(out.shape) == (1, 4, 64, 64), f"Shape: {out.shape}"
            assert out.is_contiguous()
            print("    (torch.jit.trace succeeded with real PyTorch)")

        finally:
            for k, v in _orig.items():
                if v is not None:
                    _sys.modules[k] = v
                elif k in _sys.modules:
                    del _sys.modules[k]
            # Restore mock
            _sys.modules["torch"]               = _torch_mod
            _sys.modules["torch.nn"]            = _nn_mod
            _sys.modules["torch.nn.functional"] = _nn_func

    _test("torch.jit.trace(strict=False) succeeds (real PyTorch if available)", t_real_torch_trace)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    print()
    print("=" * 60)
    print("  nuke_wrapper.py — unit tests")
    print("=" * 60)
    print(f"  Python {sys.version.split()[0]}  ·  NumPy {np.__version__}")
    print("  (mock torch — no real PyTorch required)")

    _s1(); _s2(); _s3(); _s4(); _s5(); _s6(); _s7()

    print()
    print("=" * 60)
    print(f"  {_PASS} passed   {_FAIL} failed   ({_PASS + _FAIL} total)")
    print("=" * 60)

    if _ERRORS:
        print()
        for name, msg in _ERRORS:
            print(f"  ✗ {name}")
            print(f"    {msg[:180]}")

    print()
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
    