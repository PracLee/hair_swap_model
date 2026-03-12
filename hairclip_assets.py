from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from utils.runtime_compat import safe_torch_load


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PRETRAINED_DIR = PROJECT_ROOT / "pretrained_models"
DEFAULT_LPIPS_DIR = PROJECT_ROOT / "criteria" / "lpips" / "weights"
LEGACY_BACKUP_ROOT = PROJECT_ROOT / "output"

RUNTIME_PRETRAINED_FILES = (
    "ffhq.pt",
    "seg.pth",
    "bald_proxy.pt",
    "ffhq_PCA.npz",
)
LPIPS_REQUIRED_FILES = (
    Path("v0.1") / "vgg.pth",
)


class AssetError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class AssetLayout:
    pretrained_dir: Path
    lpips_dir: Path
    files: Dict[str, Path]


@dataclasses.dataclass(frozen=True)
class PretrainedAssetSpec:
    name: str
    validator: Callable[[Path], None]
    downloadable: bool = True


def _validate_stylegan_checkpoint(path: Path) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise AssetError("StyleGAN checkpoint is not a dictionary.")
    for key in ("g_ema", "latent_avg"):
        if key not in payload:
            raise AssetError(f"Missing key: {key}")


def _validate_segmentation_checkpoint(path: Path) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or not payload:
        raise AssetError("Segmentation checkpoint is empty or invalid.")


def _validate_bald_checkpoint(path: Path) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise AssetError("Bald checkpoint is not a dictionary.")
    for key in ("alpha", "state_dict"):
        if key not in payload:
            raise AssetError(f"Missing key: {key}")


def _validate_ffhq_pca(path: Path) -> None:
    import numpy as np

    payload = np.load(path)
    for key in ("X_mean", "X_comp", "X_stdev"):
        if key not in payload:
            raise AssetError(f"Missing key: {key}")


def _validate_lpips_vgg(path: Path) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise AssetError("LPIPS weight file is not a dictionary.")
    if "lin0.model.1.weight" not in payload:
        raise AssetError("Missing LPIPS VGG weights.")


PRETRAINED_SPECS: Dict[str, PretrainedAssetSpec] = {
    "ffhq.pt": PretrainedAssetSpec("ffhq.pt", validator=_validate_stylegan_checkpoint),
    "seg.pth": PretrainedAssetSpec("seg.pth", validator=_validate_segmentation_checkpoint),
    "bald_proxy.pt": PretrainedAssetSpec("bald_proxy.pt", validator=_validate_bald_checkpoint),
    "ffhq_PCA.npz": PretrainedAssetSpec(
        "ffhq_PCA.npz",
        validator=_validate_ffhq_pca,
        downloadable=False,
    ),
}


def _unique_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        yield resolved


def iter_pretrained_filenames() -> Iterable[str]:
    yield from RUNTIME_PRETRAINED_FILES


def _iter_legacy_pretrained_dirs() -> Iterable[Path]:
    if not LEGACY_BACKUP_ROOT.is_dir():
        return
    for backup_dir in sorted(LEGACY_BACKUP_ROOT.glob("runpod-backup-*"), reverse=True):
        candidate = backup_dir / "pretrained_models"
        if candidate.is_dir():
            yield candidate


def _iter_legacy_lpips_dirs() -> Iterable[Path]:
    if not LEGACY_BACKUP_ROOT.is_dir():
        return
    for backup_dir in sorted(LEGACY_BACKUP_ROOT.glob("runpod-backup-*"), reverse=True):
        candidate = backup_dir / "lpips_weights"
        if candidate.is_dir():
            yield candidate


def candidate_pretrained_dirs() -> Iterable[Path]:
    env_dir = os.environ.get("HAIRCLIP_PRETRAINED_DIR")
    asset_root = os.environ.get("HAIRCLIP_ASSET_ROOT")
    paths = []
    if env_dir:
        paths.append(Path(env_dir))
    if asset_root:
        paths.append(Path(asset_root) / "pretrained_models")
    paths.append(DEFAULT_PRETRAINED_DIR)
    paths.extend(_iter_legacy_pretrained_dirs())
    yield from _unique_paths(paths)


def candidate_lpips_dirs() -> Iterable[Path]:
    env_dir = os.environ.get("HAIRCLIP_LPIPS_DIR")
    asset_root = os.environ.get("HAIRCLIP_ASSET_ROOT")
    paths = []
    if env_dir:
        paths.append(Path(env_dir))
    if asset_root:
        paths.append(Path(asset_root) / "lpips_weights")
    paths.append(DEFAULT_LPIPS_DIR)
    paths.extend(_iter_legacy_lpips_dirs())
    yield from _unique_paths(paths)


def resolve_pretrained_file(filename: str) -> Optional[Path]:
    for directory in candidate_pretrained_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_lpips_dir() -> Optional[Path]:
    for directory in candidate_lpips_dirs():
        if all((directory / relative_path).is_file() for relative_path in LPIPS_REQUIRED_FILES):
            return directory
    return None


def validate_pretrained_file(filename: str, path: Optional[Path] = None) -> Optional[str]:
    asset_path = Path(path or resolve_pretrained_file(filename) or "")
    if not asset_path or not asset_path.is_file():
        return "missing"

    spec = PRETRAINED_SPECS[filename]
    try:
        spec.validator(asset_path)
    except Exception as exc:
        return f"{exc.__class__.__name__}: {exc}"
    return None


def validate_lpips_dir(path: Optional[Path] = None) -> Dict[str, str]:
    issues: Dict[str, str] = {}
    lpips_dir = Path(path or resolve_lpips_dir() or "")
    for relative_path in LPIPS_REQUIRED_FILES:
        asset_path = lpips_dir / relative_path
        if not asset_path.is_file():
            issues[str(relative_path)] = "missing"
            continue
        try:
            _validate_lpips_vgg(asset_path)
        except Exception as exc:
            issues[str(relative_path)] = f"{exc.__class__.__name__}: {exc}"
    return issues


def collect_asset_issues() -> Dict[str, str]:
    issues: Dict[str, str] = {}
    for filename in iter_pretrained_filenames():
        issue = validate_pretrained_file(filename)
        if issue is not None:
            issues[filename] = issue
    for relative_path, issue in validate_lpips_dir().items():
        normalized = Path(relative_path).as_posix()
        issues[f"criteria/lpips/weights/{normalized}"] = issue
    return issues


def restore_legacy_assets(
    target_pretrained_dir: Optional[Path] = None,
    target_lpips_dir: Optional[Path] = None,
) -> Dict[str, int]:
    target_pretrained_dir = Path(target_pretrained_dir or DEFAULT_PRETRAINED_DIR)
    target_lpips_dir = Path(target_lpips_dir or DEFAULT_LPIPS_DIR)
    copied_pretrained = 0
    copied_lpips = 0
    expected_pretrained = set(iter_pretrained_filenames())

    legacy_pretrained = next(iter(_iter_legacy_pretrained_dirs() or ()), None)
    if legacy_pretrained is not None:
        target_pretrained_dir.mkdir(parents=True, exist_ok=True)
        for source_path in legacy_pretrained.iterdir():
            if not source_path.is_file() or source_path.name not in expected_pretrained:
                continue
            target_path = target_pretrained_dir / source_path.name
            if target_path.is_file() and target_path.stat().st_size == source_path.stat().st_size:
                continue
            shutil.copy2(source_path, target_path)
            copied_pretrained += 1

    legacy_lpips = next(iter(_iter_legacy_lpips_dirs() or ()), None)
    if legacy_lpips is not None:
        target_lpips_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in LPIPS_REQUIRED_FILES:
            source_path = legacy_lpips / relative_path
            if not source_path.is_file():
                continue
            target_path = target_lpips_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_file() and target_path.stat().st_size == source_path.stat().st_size:
                continue
            shutil.copy2(source_path, target_path)
            copied_lpips += 1

    return {
        "copied_pretrained": copied_pretrained,
        "copied_lpips": copied_lpips,
    }


def ensure_runtime_assets(
    restore_from_legacy: bool = True,
    validate: bool = False,
) -> AssetLayout:
    if restore_from_legacy:
        restore_legacy_assets()

    resolved_files: Dict[str, Path] = {}
    missing_files = []
    for filename in iter_pretrained_filenames():
        path = resolve_pretrained_file(filename)
        if path is None:
            missing_files.append(filename)
        else:
            resolved_files[filename] = path

    lpips_dir = resolve_lpips_dir()
    if lpips_dir is None:
        missing_files.append(f"criteria/lpips/weights/{LPIPS_REQUIRED_FILES[0]}")

    if missing_files:
        searched_pretrained = ", ".join(str(path) for path in candidate_pretrained_dirs())
        searched_lpips = ", ".join(str(path) for path in candidate_lpips_dirs())
        raise AssetError(
            "Missing runtime assets: "
            + ", ".join(missing_files)
            + f". searched_pretrained=[{searched_pretrained}]"
            + f" searched_lpips=[{searched_lpips}]"
        )

    if validate:
        issues = collect_asset_issues()
        if issues:
            formatted = ", ".join(f"{name}: {issue}" for name, issue in issues.items())
            raise AssetError(f"Invalid runtime assets: {formatted}")

    return AssetLayout(
        pretrained_dir=resolved_files["ffhq.pt"].parent,
        lpips_dir=lpips_dir,
        files=resolved_files,
    )
