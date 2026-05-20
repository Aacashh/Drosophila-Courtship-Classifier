"""Cached-features workflow for interactive heuristic threshold tuning.

The expensive part of behavior classification is feature extraction (head/tail
resolution, kinematic windowing, social features). Once those exist, applying
threshold rules is cheap. This module caches the feature payload per chamber
so sliders can re-score in real time without re-running the tracker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from classification.heuristic import (
    DEFAULT_PARAMS,
    _apply_rules,
    _compute_features,
)
from utils.export import infer_sex


# Parameters surfaced as interactive sliders. Chosen because they dominate
# the unified Courtship label.
SLIDER_PARAMS: List[Tuple[str, float, float, float]] = [
    # (name, min, max, step)
    ("wing_ext_angle_deg", 30.0, 90.0, 1.0),
    ("follow_velocity_mm_s", 0.2, 5.0, 0.1),
    ("follow_dist_max_mm", 4.0, 20.0, 0.5),
    ("follow_facing_deg", 30.0, 120.0, 1.0),
    ("circling_lat_vel_mm_s", 0.2, 5.0, 0.1),
    ("att_cop_nose_dist_mm", 0.2, 3.0, 0.1),
    ("cop_distance_mm", 1.0, 8.0, 0.1),
    ("merge_gap_s", 0.0, 5.0, 0.1),
    ("courtship_merge_gap_s", 0.0, 10.0, 0.5),
    ("courtship_min_dur_s", 0.0, 5.0, 0.1),
]


@dataclass
class CachedFeatures:
    """Tracked + featurized chamber data ready for fast re-scoring."""
    chamber_idx: int
    fps: float
    features: Dict
    tracks: Dict
    px_per_mm: float


def build_cache(
    chamber_idx: int,
    tracks: Dict,
    fps: float,
    px_per_mm: float,
) -> CachedFeatures:
    """Run the slow feature step once and return a cache entry."""
    features = _compute_features(tracks, fps)
    return CachedFeatures(
        chamber_idx=chamber_idx,
        fps=fps,
        features=features,
        tracks=tracks,
        px_per_mm=px_per_mm,
    )


def rescore(
    cached: CachedFeatures,
    params: Optional[Dict] = None,
    px_per_mm_override: Optional[float] = None,
) -> Dict[str, List[List[Tuple[int, int]]]]:
    """Re-run threshold rules against cached features."""
    scale = px_per_mm_override if px_per_mm_override is not None else cached.px_per_mm
    return _apply_rules(cached.features, cached.fps, scale, params=params)


def courtship_bouts_seconds(
    bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
    fps: float,
    male_only: bool = True,
    tracks: Optional[Dict] = None,
) -> Tuple[List[Tuple[float, float]], str]:
    """Pull unified Courtship bouts (in seconds) from rescore output.

    Returns (bouts, scope) where scope is 'male' / 'union' / 'empty'.
    Sex inference uses the same multi-signal rule as the export pipeline.
    """
    courtship = bouts_by_behavior.get("Courtship", [[], []])
    if not courtship:
        return [], "empty"

    male_idx: Optional[int] = None
    if male_only:
        sex = infer_sex(bouts_by_behavior, tracks)
        for fly_idx, info in sex.items():
            if info.get("is_male"):
                male_idx = fly_idx
                break

    if male_idx is not None and male_idx < len(courtship):
        bouts = [(s / fps, (e + 1) / fps) for s, e in courtship[male_idx]]
        return sorted(bouts), "male"

    # Union both flies
    merged: List[Tuple[float, float]] = []
    raw: List[Tuple[float, float]] = []
    for fly_bouts in courtship:
        for s, e in fly_bouts:
            raw.append((s / fps, (e + 1) / fps))
    raw.sort()
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, "union"


# ---------------------------------------------------------------------------
# Params persistence
# ---------------------------------------------------------------------------


def load_tuned_params(path: Optional[Path]) -> Dict:
    """Load `tuned_params.json` and merge over `DEFAULT_PARAMS`."""
    if path is None or not Path(path).exists():
        return dict(DEFAULT_PARAMS)
    try:
        overrides = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_PARAMS)
    merged = dict(DEFAULT_PARAMS)
    merged.update(overrides)
    return merged


def save_tuned_params(params: Dict, path: Path) -> None:
    """Write only the keys that differ from DEFAULT_PARAMS to keep diffs small."""
    diff = {k: v for k, v in params.items()
            if k in DEFAULT_PARAMS and v != DEFAULT_PARAMS[k]}
    Path(path).write_text(json.dumps(diff, indent=2), encoding="utf-8")


def diff_from_default(params: Dict) -> Dict:
    """Return the subset of `params` that differs from `DEFAULT_PARAMS`."""
    return {k: v for k, v in params.items()
            if k in DEFAULT_PARAMS and v != DEFAULT_PARAMS[k]}


# ---------------------------------------------------------------------------
# Tracks persistence (per-chamber NPZ next to the results CSV)
# ---------------------------------------------------------------------------


_TRACK_FIELDS = ("x", "y", "theta", "a", "b", "area", "wing_angle")


def tracks_npz_path(results_dir: Path, chamber_idx: int) -> Path:
    return Path(results_dir) / f"chamber_{chamber_idx}_tracks.npz"


def save_tracks(tracks: Dict, fps: float, path: Path) -> None:
    """Persist a `tracks` dict (and fps) to a compressed NPZ file."""
    payload: Dict[str, np.ndarray] = {"fps": np.array([fps], dtype=float)}
    for field in _TRACK_FIELDS:
        if field not in tracks:
            continue
        for fly_idx, arr in enumerate(tracks[field]):
            payload[f"fly_{fly_idx}_{field}"] = np.asarray(arr, dtype=float)
    if "overlap" in tracks:
        overlap = tracks["overlap"]
        if isinstance(overlap, list):
            for i, arr in enumerate(overlap):
                payload[f"overlap_{i}"] = np.asarray(arr, dtype=bool)
        else:
            payload["overlap_0"] = np.asarray(overlap, dtype=bool)
    np.savez_compressed(path, **payload)


def load_tracks(path: Path) -> Optional[Tuple[Dict, float]]:
    """Reload a tracks dict + fps from NPZ. Returns None if file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except (OSError, ValueError):
        return None

    fps = float(data["fps"][0]) if "fps" in data.files else 30.0
    tracks: Dict[str, List[np.ndarray]] = {f: [] for f in _TRACK_FIELDS}
    for field in _TRACK_FIELDS:
        for fly_idx in (0, 1):
            key = f"fly_{fly_idx}_{field}"
            if key in data.files:
                tracks[field].append(np.asarray(data[key]))
    overlap = [np.asarray(data[k]) for k in data.files if k.startswith("overlap_")]
    if overlap:
        tracks["overlap"] = overlap
    return tracks, fps
