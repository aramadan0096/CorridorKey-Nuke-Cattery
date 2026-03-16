# CorridorKey — Nuke Cattery Integration

Native **Foundry Nuke 17.0+** support for [CorridorKey](CorridorKey\README.md).  
Run the full green-screen unmixing network as a live Inference node directly inside your comp, using any existing Nuke keying node as the alpha hint input.

![screenshot](assets\nuke_cattery.png)

---

## What this adds

Four files inside this `nuke/` folder are all you need:

```
nuke/
├── nuke_wrapper.py          Core — TorchScript wrapper around GreenFormer
├── export_torchscript.py    Run this once to produce CorridorKey.pt
├── download_checkpoint.py   Reliably downloads the real ~300 MB weights
├── test_nuke_wrapper.py     32 self-contained tests (no PyTorch required)
├── inspect_model.py         Diagnostic tool if anything goes wrong
├── Template.nk              ← Template Nuke script — open this in NukeX to build the .cat model
│                              generate CorridorKey.cat with one click
└── README.md                This file
```

No upstream files are modified. The `nuke/` folder is purely additive.

---

## How it works

### Architecture

The CorridorKey model (`GreenFormer`) is a two-stage encoder-decoder:

<!-- ```
Input  [1, 4, H, W]
  ch 0-2  plate RGB      sRGB or linear EXR, values [0, 1]
  ch 3    alpha hint     coarse mask from any Nuke keying node, [0, 1]
        │
        ▼  resize to 2048×2048 (model native resolution)
        │
  ┌─────────────────────────────────────────────┐
  │  Encoder   — ViT, 24 transformer blocks     │
  │  (pos_embed, patch_embed kernel=7 stride=4) │
  └──────────────┬──────────────────────────────┘
                 │  multiscale feature maps
       ┌─────────┴──────────┐
  Alpha decoder          FG decoder
  (SegFormer MLP head)   (SegFormer MLP head)
  → coarse alpha         → coarse FG colour
       └─────────┬──────────┘
                 │
          CNN Refiner
          (stem + res1-4 + final)
          → fine alpha + fine FG
        │
        ▼  blend coarse→fine by refiner_strength knob
        ▼  resize back to original plate resolution
        │
Output [1, 4, H, W]
  ch 0-2  straight sRGB foreground   [0, 1]
  ch 3    refined linear alpha matte [0, 1]
``` -->

![flowchart diagram](assets\flowchart_diagram.png)

### Why `torch.jit.trace()` and not `torch.jit.script()`

The ViT backbone uses `timm` internally, which relies on dynamic Python registries incompatible with TorchScript's static analysis. `torch.jit.trace(model, dummy, strict=False)` records the concrete computation graph for a single forward pass — no dynamic code needed. The wrapper handles all input resolutions via `F.interpolate`, so Nuke can send any plate size and receive a correctly-sized output.

### Three non-obvious checkpoint problems solved

| Problem | Symptom | Fix |
|---|---|---|
| Checkpoint saved from `torch.compile()` | All state-dict keys carry `_orig_mod.` prefix | Strip prefix before `load_state_dict` |
| Model class name not documented | `ImportError: cannot import name 'CorridorKeyTransformer'` | Discover by sub-module matching, not by name |
| `GreenFormer` default `img_size` is wrong | `size mismatch for pos_embed` | Read actual patch **stride** from a live instance, compute `img_size = sqrt(num_patches) × stride` |

---

## Step-by-step setup

#### 0. Clone repository:

```sh
git clone --recurse-submodules https://github.com/aramadan0096/CorridorKey-Nuke-Cattery
```

#### Bootstrap

Download and install uv package manager via winget and install python and dependencies.

```sh
.\install.bat
```
Download CorridorKey checkpoint and stat exporting torch script to `Export` folder.

```sh
.\start.bat
```

### 1. Download the real weights

The standard `hf download` command only downloads Git LFS pointers on Windows without a token. Use the dedicated script instead:

```powershell
uv run python nuke/download_checkpoint.py
```

This places `CorridorKey.pth` (~300 MB) in `CorridorKeyModule/checkpoints/`.

### 2. Run the tests (optional but recommended)

No PyTorch, GPU, or checkpoint required — runs on pure NumPy in seconds:

```powershell
python nuke/test_nuke_wrapper.py
# Expected: 32 passed, 0 failed
```

### 3. Export to TorchScript

```powershell
uv run python nuke/export_torchscript.py
```

This produces `nuke/CorridorKey.pt` (~300 MB). The script validates shape, range, contiguity, and hint response before saving, and will abort with a clear error if the checkpoint is missing or corrupt.

On success you will see:
```
  [CorridorKey] Stripped '_orig_mod.' prefix from 367 keys
  [CorridorKey] Inferred img_size=2048 from patch stride=4
  [CorridorKey] Selected 'GreenFormer' (score 367/367, ~300 M params)
  [CorridorKey] Loaded 300 MB checkpoint → 'GreenFormer'
  ✓ Trace complete
► Saved: nuke\CorridorKey.pt  (300 MB)
```

### 4. Generate the `.cat` file in NukeX

Open the provided template script in NukeX 17.0:

```
File → Import Script → nuke/CatFileCreators.nk
```

This loads a pre-configured **CatFileCreator** node. Update the two paths to point to your local files, then click **"Create .cat file and Inference"**:

| Field | Value |
|---|---|
| Torchscript File | `/your/path/nuke/CorridorKey.pt` |
| Cat File | `/your/path/nuke/CorridorKey.cat` |
| Channels In | `rgba.red, rgba.green, rgba.blue, rgba.alpha` |
| Channels Out | `rgba.red, rgba.green, rgba.blue, rgba.alpha` |
| Model Id | `CorridorKey_v1.0_Nuke` |
| Output Scale | `1` |

The three custom knobs are already defined in the template:

| Knob | Type | Default | Range | What it does |
|---|---|---|---|---|
| `despill_strength` | Float | `0.0` | 0 – 10 | Blends green channel toward avg(R, B). 0 = off. |
| `gamma_input` | Enum | `sRGB` | sRGB / Linear | Declare your plate's gamma encoding. |
| `refiner_strength` | Float | `1.0` | 0 – 1 | 0 = coarse prediction only. 1 = full CNN refinement. |

> **Important:** The `Name` field of each knob in CatFileCreator must match the Python attribute name character-for-character. The template has this set correctly already.

---

## Nuke compositing setup

<!-- ```
┌─ Read ─────────────────┐   ┌─ Read ─────────────────────────────────────┐
│ green_screen_plate.exr  │   │ alpha_hint (any source — see below)        │
│                         │   │                                            │
│ If sRGB PNG/TIFF:       │   │ • IBK / Primatte rough key                 │
│   check Raw Data        │   │ • Keylight core matte                      │
│   gamma_input = sRGB    │   │ • Roto / RotoPaint                         │
│                         │   │ • Any node whose alpha output              │
│ If linear EXR:          │   │   isolates your subject roughly            │
│   uncheck Raw Data      │   │                                            │
│   gamma_input = Linear  │   │ Does NOT need to be clean —                │
└─────────┬───────────────┘   │ a rough eroded matte works best            │
          │                   └──────────────┬─────────────────────────────┘
          │ plate RGB                        │ hint alpha
          └──────────────┬───────────────────┘
                         │
                   Shuffle2
          plate → rgba.red / .green / .blue
          hint  → rgba.alpha
                         │
              Inference (CorridorKey.cat)
              Channels In:  rgba.red .green .blue .alpha
              Channels Out: rgba.red .green .blue .alpha
                         │
              ch 0-2 = straight sRGB FG
              ch 3   = refined linear alpha
                         │
                ┌────────┴────────────────────────┐
                │  REQUIRED — do not skip          │
                │  Colorspace (sRGB → linear)      │
                │  applied to rgb channels only    │
                └────────┬────────────────────────┘
                         │
                       Premult
                (linear FG × linear alpha)
                         │
                   Merge (Over)
                (A = premult FG, B = background)
                         │
                    Viewer / Write
``` -->

![Nuke compositing setup](assets\comp_tree.png)

> **The Colorspace node between Inference and Premult is not optional.**  
> The model outputs foreground colour in sRGB gamma. Premultiplying sRGB values against a linear alpha produces dark, crushed edges on every semi-transparent pixel — hair, motion blur, fine fabric. Always linearise the FG channels before Premult.

### Routing the hint

The hint connects to `rgba.alpha` of the Shuffle2 node. Use whichever channel your keyer produces:

| Upstream output | Shuffle2 mapping |
|---|---|
| IBK / Primatte alpha in `rgba.alpha` | Input B → out.alpha |
| Roto mask in `rgba.red` | Input B.red → out.alpha |
| Luminance of any image | Add Colorspace node first, then route .alpha |

---

## Knob usage guide

**`gamma_input`** — the most important knob to get right.

| Plate format | Read node setting | gamma_input |
|---|---|---|
| sRGB PNG, TIFF, JPEG | check Raw Data | sRGB (0) |
| Linear EXR (VFX render, ACES) | uncheck Raw Data | Linear (1) |
| Log DPX / Cineon | add Colorspace (log→sRGB) before Shuffle2 | sRGB (0) |

**`despill_strength`** — leave at 0 if you want full control downstream. For a quick deliverable, values of 3–6 remove most visible green contamination without affecting neutral tones.

**`refiner_strength`** — reduce to 0.7–0.8 if you see shimmering or over-refined edge artefacts between frames. The coarse prediction (0.0) is already clean on most subjects; the refiner adds fine hair and motion-blur detail.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Dark fringe on composite | FG not linearised before Premult | Add Colorspace (sRGB→linear) between Inference and Premult |
| Wrong gamma setting | Plate gamma declared incorrectly | Match `gamma_input` to your plate's actual encoding |
| `size mismatch for pos_embed` | Old `nuke_wrapper.py` | Update to the latest version — stride-based `img_size` inference |
| `_orig_mod.` prefix error | Old `nuke_wrapper.py` | Update — prefix stripping is now automatic |
| `.pt` is only ~10 KB | Stub model was traced, not the real network | Re-run `download_checkpoint.py`, then `export_torchscript.py` |
| Out of memory at full res | GPU VRAM insufficient | Lower plate resolution before Inference, upscale result after |
| Hint too expanded → leaking BG | Alpha hint covers too much | Erode hint 2–5 px before Shuffle2 |
| Missing fine hair detail | Alpha hint too tight | Relax erosion; try slight Gaussian blur on hint |

**Diagnostic tool** — if the export crashes with an unexpected class or size error:

```powershell
uv run python nuke/inspect_model.py
```

This prints every `nn.Module` class in `model_transformer.py`, its sub-modules, the checkpoint's top-level structure, and which class the discovery algorithm would select.

---

## File reference

| File | Purpose |
|---|---|
| `nuke_wrapper.py` | `CorridorKeyNukeWrapper(nn.Module)` — the TorchScript model. Contains `_strip_orig_mod()`, `_discover_model_class()`, `_read_patch_stride()`, and the `forward()` method that Nuke calls per frame. |
| `export_torchscript.py` | Loads the wrapper, traces it at 2048×2048, validates output, saves `CorridorKey.pt`. Run once after downloading weights. |
| `download_checkpoint.py` | Downloads `CorridorKey.pth` via `hf_hub_download()` (handles LFS correctly on Windows) with a direct-URL fallback. |
| `CatFileCreators.nk` | Template Nuke script. Open in NukeX 17.0 to generate `CorridorKey.cat` with one click. Paths need updating to your local locations. |
| `test_nuke_wrapper.py` | 32 self-contained tests covering sRGB math, despill, I/O contract, knob types, gamma routing, and channel order. Runs without PyTorch via a NumPy mock. |
| `inspect_model.py` | Diagnostic: lists all classes in `model_transformer.py` and shows which one would be selected for the current checkpoint. |

---

## Licensing

This Nuke integration inherits the CorridorKey license (CC BY-NC-SA 4.0 variant).

- ✅ Use inside commercial VFX productions as part of your in-house pipeline  
- ✅ Distribute `CorridorKey.cat` internally within your studio  
- ❌ Sell as a commercial plugin or product  
- ❌ Offer as a paid cloud inference service  

<!-- For commercial plugin integration: `contact@corridordigital.com` -->

Nuke Cattery Implementation Built on [CorridorKey](CorridorKey\README.md) Authored by [Ahmed Ramadan](https://github.com/aramadan0096)