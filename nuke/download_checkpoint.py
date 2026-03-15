#!/usr/bin/env python3
###############################################################################
# CorridorKey for Nuke
# Authored by: Ahmed Ramadan
# This software is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# Full license text: https://creativecommons.org/licenses/by-nc-nd/4.0
# Repository: https://github.com/nikopueringer/CorridorKey
###############################################################################

"""
download_checkpoint.py
======================
Downloads the CorridorKey v1.0 weights binary (~300 MB) into
CorridorKeyModule/checkpoints/CorridorKey.pth

Run once from the CorridorKey repo root before exporting:

    uv run python nuke/download_checkpoint.py

Why not just 'hf download nikopueringer/CorridorKey_v1.0'?
-----------------------------------------------------------
The 'hf download' (huggingface-hub CLI) command downloads ALL files in
a repo.  The .pth file is stored in Git LFS.  Without an authenticated
HF_TOKEN, or on Windows without symlink support, the CLI may download
only the small LFS pointer text file (~134 bytes) instead of the actual
300 MB binary — giving you a checkpoint that appears to exist but fails
silently when PyTorch tries to unpickle it.

This script uses hf_hub_download() which resolves and downloads the real
binary directly to the local path, bypassing the LFS pointer issue.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

# Repo root = parent of nuke/
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_DEST       = _REPO_ROOT / "CorridorKeyModule" / "checkpoints" / "CorridorKey.pth"
_HF_REPO    = "nikopueringer/CorridorKey_v1.0"
_HF_FILE    = "CorridorKey_v1.0.pth"
_DIRECT_URL = (
    "https://huggingface.co/nikopueringer/CorridorKey_v1.0"
    "/resolve/main/CorridorKey_v1.0.pth"
)
_MIN_SIZE   = 200_000_000   # 200 MB minimum for a real checkpoint


def _size_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= _MIN_SIZE


def _progress(block_num: int, block_size: int, total: int) -> None:
    downloaded = block_num * block_size
    if total > 0:
        pct = min(100.0, downloaded / total * 100)
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"{downloaded/1e6:6.0f} / {total/1e6:.0f} MB",
            end="", flush=True,
        )


def _download_via_hf_hub() -> bool:
    """Use huggingface_hub.hf_hub_download (handles auth + LFS correctly)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    print(f"  Trying huggingface_hub.hf_hub_download …")
    try:
        cached = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_FILE,
        )
        src = Path(cached)
        if src.stat().st_size < _MIN_SIZE:
            print(f"  ✗ Downloaded file too small ({src.stat().st_size:,} bytes) — "
                  "may be an LFS pointer. Trying direct URL …")
            return False

        _DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(_DEST))
        return True

    except Exception as e:
        print(f"  ✗ hf_hub_download failed: {e}")
        return False


def _download_direct() -> bool:
    """Fall back to a direct HTTPS download of the LFS binary URL."""
    print(f"  Trying direct URL download …")
    print(f"  URL: {_DIRECT_URL}")
    try:
        _DEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DEST.with_suffix(".tmp")
        urllib.request.urlretrieve(_DIRECT_URL, str(tmp), reporthook=_progress)
        print()  # newline after progress bar

        actual_size = tmp.stat().st_size
        if actual_size < _MIN_SIZE:
            print(f"  ✗ Downloaded {actual_size:,} bytes — still too small.")
            print("    The file may be an LFS redirect requiring authentication.")
            print("    Set HF_TOKEN in your environment and retry.")
            tmp.unlink(missing_ok=True)
            return False

        shutil.move(str(tmp), str(_DEST))
        return True

    except Exception as e:
        print(f"  ✗ Direct download failed: {e}")
        return False


def main() -> None:
    print()
    print("=" * 62)
    print("  CorridorKey — checkpoint downloader")
    print("=" * 62)
    print(f"  Destination: {_DEST}")
    print()

    # Already downloaded and valid?
    if _size_ok(_DEST):
        size_mb = _DEST.stat().st_size / 1e6
        print(f"  ✓ Already present ({size_mb:.0f} MB) — nothing to do.")
        print()
        return

    # Is there a small/broken file to diagnose?
    if _DEST.exists():
        size = _DEST.stat().st_size
        hint = ""
        try:
            with open(_DEST, "rb") as f:
                header = f.read(200).decode("utf-8", errors="replace")
            if "git-lfs" in header or "oid sha256" in header:
                hint = " (Git LFS pointer — not the real binary)"
        except Exception:
            pass
        print(f"  ⚠ Existing file is only {size:,} bytes{hint}. Replacing it.")
        _DEST.unlink()

    # Try method 1: huggingface_hub
    ok = _download_via_hf_hub()

    # Try method 2: direct URL
    if not ok:
        ok = _download_direct()

    if not ok:
        print()
        print("  Both download methods failed.")
        print()
        print("  Manual options:")
        print(f"  1. Set HF_TOKEN and retry:")
        print(f"       set HF_TOKEN=hf_xxxx   (Windows)")
        print(f"       export HF_TOKEN=hf_xxxx (Linux/Mac)")
        print(f"       uv run python nuke/download_checkpoint.py")
        print()
        print(f"  2. Download in a browser:")
        print(f"       {_DIRECT_URL}")
        print(f"     Save as: {_DEST}")
        sys.exit(1)

    # Final verification
    final_size = _DEST.stat().st_size
    print()
    print(f"  ✓ Saved: {_DEST}  ({final_size/1e6:.0f} MB)")
    print()
    print("  Next step:")
    print("    uv run python nuke/export_torchscript.py")
    print()


if __name__ == "__main__":
    main()