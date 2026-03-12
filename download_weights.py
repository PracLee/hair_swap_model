#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict

from hairclip_assets import (
    LPIPS_REQUIRED_FILES,
    RUNTIME_PRETRAINED_FILES,
    AssetError,
    DEFAULT_LPIPS_DIR,
    DEFAULT_PRETRAINED_DIR,
    PRETRAINED_SPECS,
    collect_asset_issues,
    ensure_runtime_assets,
    restore_legacy_assets,
)
from utils.env_loader import load_project_dotenv
from utils.sam2_runtime import maybe_download_sam2_checkpoint, resolve_sam2_checkpoint_path


load_project_dotenv()


GDRIVE_DOWNLOADS: Dict[str, str] = {
    "ffhq.pt": "1g8S81ZybmrF86OjvjLYJzx-wx83ZOiIw",
    "seg.pth": "1OG6t7q4PpHOoYNdP-ipoxuqYbfMSgPta",
    "bald_proxy.pt": "1sa732uBfX1739MFsvtRCKWCN54zYyltC",
}

LPIPS_DOWNLOADS: Dict[Path, str] = {
    Path("v0.1") / "vgg.pth": "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth",
}


def _ensure_gdown() -> object:
    try:
        return importlib.import_module("gdown")
    except ImportError as exc:
        raise RuntimeError(
            "gdown is not installed. Install requirements.txt first or rerun install_requirements.sh."
        ) from exc


def _download_pretrained_file(filename: str, target_dir: Path) -> bool:
    file_id = GDRIVE_DOWNLOADS.get(filename)
    if not file_id:
        return False
    gdown = _ensure_gdown()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if target_path.exists():
        target_path.unlink()
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(target_path), quiet=False)
    return target_path.is_file() and target_path.stat().st_size > 0


def _download_lpips_file(relative_path: Path, target_dir: Path) -> bool:
    url = LPIPS_DOWNLOADS.get(relative_path)
    if not url:
        return False
    target_path = target_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    with urllib.request.urlopen(url, timeout=30) as response:
        target_path.write_bytes(response.read())
    return target_path.is_file() and target_path.stat().st_size > 0


def _print_status() -> None:
    requested = list(RUNTIME_PRETRAINED_FILES)
    print("[asset-bootstrap] checking runtime assets")
    print(f"  pretrained_dir: {DEFAULT_PRETRAINED_DIR}")
    print(f"  lpips_dir:      {DEFAULT_LPIPS_DIR}")
    print(f"  pretrained:     {requested}")
    print(f"  lpips:          {[str(path) for path in LPIPS_REQUIRED_FILES]}")
    print(f"  sam2_checkpoint:{resolve_sam2_checkpoint_path()}")


def _repair_sam2(allow_missing_downloads: bool, checkpoint_path: str | None) -> None:
    if not allow_missing_downloads:
        return
    resolved = maybe_download_sam2_checkpoint(checkpoint_path=checkpoint_path, auto_download=True)
    if resolved is not None:
        print(f"[asset-bootstrap] restored SAM2 checkpoint: {resolved}")


def _repair_assets(allow_missing_downloads: bool, allow_corrupt_repairs: bool) -> None:
    issues = collect_asset_issues()
    if not issues:
        return

    for name, issue in issues.items():
        is_missing = issue == "missing"
        is_lpips = name.startswith("criteria/lpips/weights/")

        if is_missing and not allow_missing_downloads:
            continue
        if not is_missing and not allow_corrupt_repairs:
            continue

        if is_lpips:
            relative_path = Path(name.replace("criteria/lpips/weights/", "", 1))
            if _download_lpips_file(relative_path, DEFAULT_LPIPS_DIR):
                print(f"[asset-bootstrap] restored {name}")
            continue

        spec = PRETRAINED_SPECS[name]
        if not spec.downloadable:
            continue
        if _download_pretrained_file(name, DEFAULT_PRETRAINED_DIR):
            print(f"[asset-bootstrap] restored {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore or download runtime assets for RunPod execution")
    parser.add_argument(
        "--restore-legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy assets from output/runpod-backup-* into the root layout first.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing runtime assets after legacy restore.",
    )
    parser.add_argument(
        "--repair-corrupt",
        action="store_true",
        help="Redownload any corrupted runtime asset that has a known download source.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if required runtime assets are still missing or invalid.",
    )
    parser.add_argument(
        "--download-sam2",
        action="store_true",
        help="Download the optional SAM2 checkpoint via Hugging Face when missing.",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=None,
        help="Target path for the optional SAM2 checkpoint.",
    )
    args = parser.parse_args()

    _print_status()

    if args.restore_legacy:
        summary = restore_legacy_assets()
        print(
            "[asset-bootstrap] restored from legacy backup: "
            f"pretrained={summary['copied_pretrained']}, lpips={summary['copied_lpips']}"
        )

    _repair_assets(allow_missing_downloads=args.download_missing, allow_corrupt_repairs=args.repair_corrupt)
    _repair_sam2(allow_missing_downloads=args.download_sam2, checkpoint_path=args.sam2_checkpoint)

    issues = collect_asset_issues()
    if issues:
        print("[asset-bootstrap] unresolved issues:")
        for name, issue in sorted(issues.items()):
            print(f"  - {name}: {issue}")
        print(
            "[asset-bootstrap] note: ffhq_PCA.npz has no automatic download source configured. "
            "Keep a valid copy in pretrained_models/."
        )
        return 1 if args.strict else 0

    try:
        layout = ensure_runtime_assets(restore_from_legacy=False, validate=True)
    except AssetError as exc:
        print(f"[asset-bootstrap] incomplete: {exc}")
        return 1 if args.strict else 0

    print("[asset-bootstrap] ready")
    print(f"  resolved_pretrained_dir: {layout.pretrained_dir}")
    print(f"  resolved_lpips_dir:      {layout.lpips_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
