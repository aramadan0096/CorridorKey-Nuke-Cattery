# CorridorKey — Nuke Cattery Integration

> **What this adds** to the [nikopueringer/CorridorKey](https://github.com/nikopueringer/CorridorKey)
> repository: a single `nuke/` folder containing everything needed to export
> `CorridorKey.pt` and generate `CorridorKey.cat` for Nuke's Inference node.

---

## Folder structure

```
CorridorKey/                        ← upstream repo root (unchanged)
│
├── CorridorKeyModule/
│   ├── core/
│   │   └── model_transformer.py   ← upstream — CorridorKeyTransformer
│   ├── inference_engine.py        ← upstream — CLI inference
│   └── checkpoints/
│       └── CorridorKey.pth        ← downloaded weights (~300 MB)
│
└── nuke/                          ← NEW — everything added by this integration
    ├── nuke_wrapper.py            ← TorchScript wrapper  (core deliverable)
    ├── export_torchscript.py      ← export script → CorridorKey.pt
    ├── test_nuke_wrapper.py       ← self-contained tests (no PyTorch needed)
    └── ARCHITECTURE.md            ← this file
```

No upstream file is modified. The `nuke/` folder is purely additive.

---

## Neural network architecture

```
INPUT  [1, 4, H, W]                        (from Nuke Inference node)
  ch 0  rgba.red   ─┐
  ch 1  rgba.green  ├─ green-screen plate   sRGB or linear, [0, 1]
  ch 2  rgba.blue  ─┘
  ch 3  rgba.alpha ──  coarse alpha hint    any Nuke keying node output
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  gamma_input = 1?                       │
  │  → apply linear → sRGB encoding         │  (model trained on sRGB patches)
  └─────────────────────────────────────────┘
         │
         ▼
  F.interpolate → [1, 4, 2048, 2048]        (model's native resolution)
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  CorridorKeyTransformer                                          │
  │                                                                  │
  │  Hiera Backbone (timm, features_only=True, in_chans=4)          │
  │    Stage 0 → Stage 3   multiscale feature maps                  │
  │         │                      │                                 │
  │   Alpha Head                FG Head                             │
  │   UNet decoder              UNet decoder                        │
  │   → alpha_coarse [1,1,…]   → fg_coarse [1,3,…]                 │
  │         │                      │                                 │
  │         └──────────┬───────────┘                                 │
  │                    │   cat → [1, 7, …]                           │
  │               CNN Refiner                                        │
  │               convolutional enc-dec                              │
  │               → alpha_fine [1,1,…]  fg_fine [1,3,…]            │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  blend:  alpha = alpha_coarse + refiner_strength × (alpha_fine − alpha_coarse)
          fg    = fg_coarse    + refiner_strength × (fg_fine    − fg_coarse)
         │
         ▼
  F.interpolate → [1, 4, H, W]               (resize back to plate resolution)
         │
         ▼
  optional green despill on fg channels
         │
         ▼
OUTPUT [1, 4, H, W]
  ch 0  rgba.red   ─┐
  ch 1  rgba.green  ├─ straight sRGB foreground colour   [0, 1]
  ch 2  rgba.blue  ─┘
  ch 3  rgba.alpha ──  refined linear alpha matte         [0, 1]
```

### Why `torch.jit.trace()` and not `torch.jit.script()`

`CorridorKeyTransformer` uses the `timm` library to load the Hiera backbone.
`timm` relies on dynamic Python registries (`timm.models._registry`) and
`getattr()` lookups that the static-analysis pass of `torch.jit.script()`
cannot handle.

`torch.jit.trace(model, dummy_input, strict=False)` records the computation
graph from a single concrete forward pass, bypassing all of that.  
`strict=False` suppresses false-positive warnings from `timm` internals.

The trace is locked to the spatial resolution used at trace time (2048×2048).
Arbitrary-resolution inputs work because `F.interpolate` is called *inside*
`forward()` before and after the model — Nuke can send any plate size.

---

## Nuke .cat I/O contract

| | Nuke declaration | Meaning |
|---|---|---|
| **Channels In** | `rgba.red, rgba.green, rgba.blue, rgba.alpha` | plate RGB + coarse hint |
| **Channels Out** | `rgba.red, rgba.green, rgba.blue, rgba.alpha` | sRGB FG + linear alpha |
| **Output Scale** | `1` | output resolution = input resolution |
| **Model Id** | `CorridorKey_v1.0_Nuke` | |

> The output foreground (ch 0–2) is **sRGB-encoded**.  
> Add a **Colorspace (sRGB → linear)** node after Inference before Premult.

---

## Custom knobs

| Attribute | CatFileCreator type | Default | Range | Effect |
|---|---|---|---|---|
| `despill_strength` | `Float_Knob` | `0.0` | 0 – 10 | Blends green channel toward avg(R, B). 0 = off. |
| `gamma_input` | `Enumeration_Knob` | `0` (sRGB) | sRGB / Linear | Set to Linear for EXR plates already in linear space. |
| `refiner_strength` | `Float_Knob` | `1.0` | 0 – 1 | 0 = coarse prediction only.  1 = full CNN refinement. |

All attribute names **exactly** match the CatFileCreator `Name` fields.
No `bool` attributes — Nuke's Bool_Knob maps to `int` 0/1.

---

## Quick-start

### 1. Install dependencies

```bash
# From the CorridorKey repo root
uv sync
```

### 2. Download the checkpoint

```bash
uv run hf download nikopueringer/CorridorKey_v1.0 \
    --local-dir CorridorKeyModule/checkpoints/
mv CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth \
   CorridorKeyModule/checkpoints/CorridorKey.pth
```

### 3. Run the tests (no PyTorch or GPU needed)

```bash
python nuke/test_nuke_wrapper.py
# Expected: 32 passed, 0 failed
```

### 4. Export to TorchScript

```bash
uv run python nuke/export_torchscript.py
# Produces: nuke/CorridorKey.pt
```

With explicit options:

```bash
uv run python nuke/export_torchscript.py \
    --checkpoint CorridorKeyModule/checkpoints/CorridorKey.pth \
    --output     nuke/CorridorKey.pt \
    --validate
```

### 5. Create the .cat file in NukeX 17.0

1. Open **NukeX 17.0**
2. `Tab` → `CatFileCreator` → Enter
3. Fill in the Properties panel:

   | Field | Value |
   |---|---|
   | Torchscript File | `/absolute/path/to/nuke/CorridorKey.pt` |
   | Cat File | `/absolute/path/to/nuke/CorridorKey.cat` |
   | Channels In | `rgba.red, rgba.green, rgba.blue, rgba.alpha` |
   | Channels Out | `rgba.red, rgba.green, rgba.blue, rgba.alpha` |
   | Model Id | `CorridorKey_v1.0_Nuke` |
   | Output Scale | `1` |

4. Click the **pencil icon** (Edit User Knobs) and add:

   | Type | Name | Label | Default | Range |
   |---|---|---|---|---|
   | Float_Knob | `despill_strength` | Despill Strength | `0` | 0 – 10 |
   | Enumeration_Knob | `gamma_input` | Input Gamma | `0` | sRGB; Linear |
   | Float_Knob | `refiner_strength` | Refiner Strength | `1` | 0 – 1 |

5. Click **"Create .cat file and Inference"**

---

## Nuke compositing setup

```
Read (green-screen plate)
  └─ if sRGB PNG/TIFF → check "Raw Data", set gamma_input = sRGB
  └─ if linear EXR   → uncheck "Raw Data", set gamma_input = Linear
          │
Read (alpha hint — any keying node output with rgba)
          │
          ├──────────────────────────┐
                                     │
                              Shuffle2
                   plate → rgba.red / .green / .blue
                   hint  → rgba.alpha   (from hint's .alpha or .red channel)
                                     │
                              Inference
                           (CorridorKey.cat)
                          Channels In:  rgba.red .green .blue .alpha
                          Channels Out: rgba.red .green .blue .alpha
                                     │
                          ┌──────────┴───────────┐
                          │                      │
                    rgba.rgb (sRGB FG)     rgba.alpha (linear matte)
                          │
                   Colorspace
                  sRGB → linear          ← REQUIRED before Premult
                          │
                        Premult           ← linear FG × linear alpha
                          │
                      Merge (Over)        ← A: FG,  B: background
                          │
                        Viewer / Write
```

> ⚠ **The Colorspace node between Inference and Premult is not optional.**
> CorridorKey's foreground output is sRGB-encoded. Premultiplying sRGB values
> against a linear alpha produces dark fringes on every semi-transparent edge.

---

## Routing the alpha hint

The hint input to the Inference node is `rgba.alpha`.
Use a **Shuffle2** node to route the alpha from whichever channel your
keying tree produces:

| Upstream node output | Shuffle2 mapping |
|---|---|
| IBK / Primatte alpha in `rgba.alpha` | `B.alpha → out.alpha` |
| Roto / RotoPaint mask in `rgba.red` | `B.red → out.alpha` |
| Any node's luminance | Add a `Colorspace` (colour → luminance) before Shuffle2 |

The hint does not need to be precise.  A rough, slightly eroded matte works
best — the model fills in all the fine edge detail.

---

## Downstream pass extraction

The Inference node outputs a single 4-channel image.
To split it into separate passes for multi-layer EXR rendering:

```
Inference output
      │
 ┌────┴────┐
 │  Copy   │   from: rgba.alpha   to: matte.alpha
 └────┬────┘
      │
 ┌────┴────┐
 │  Copy   │   from: rgba.red/green/blue   to: FG.red/green/blue
 └────┬────┘
      │
  Write (multi-layer EXR)
```

Alternatively: connect the Inference output directly to the standard
`Colorspace → Premult → Merge` chain and composite inline.

---

## Licensing

CorridorKey is released under a variant of **CC BY-NC-SA 4.0**.

- ✅ Use in commercial VFX productions (internal pipeline)
- ✅ Distribute internally within your studio
- ❌ Sell as a commercial plugin or product
- ❌ Offer as a paid cloud/API inference service

For commercial software integration:
`contact@corridordigital.com`
