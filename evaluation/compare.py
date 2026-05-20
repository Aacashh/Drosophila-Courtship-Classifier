"""Compare predicted courtship bouts against the reference annotations.

Two metric families:
- Frame-level (precision / recall / F1 / IoU): every frame is a binary label,
  treats over- and under-detection symmetrically.
- Bout-level (matched / missed / false positive): a bout in the prediction is
  matched to a reference bout if their temporal IoU exceeds a threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .reference_parser import ReferenceChamber


Bout = Tuple[float, float]


# ---------------------------------------------------------------------------
# Frame-level metrics
# ---------------------------------------------------------------------------


def bouts_to_frame_labels(
    bouts: Sequence[Bout],
    n_frames: int,
    fps: float,
) -> np.ndarray:
    """Convert (start_s, end_s) bouts into a boolean array of length n_frames."""
    labels = np.zeros(n_frames, dtype=bool)
    if fps <= 0 or n_frames <= 0:
        return labels
    for start_s, end_s in bouts:
        start_f = max(0, int(math.floor(start_s * fps)))
        end_f = min(n_frames, int(math.ceil(end_s * fps)))
        if end_f > start_f:
            labels[start_f:end_f] = True
    return labels


@dataclass
class FrameMetrics:
    precision: float
    recall: float
    f1: float
    iou: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "frame_precision": self.precision,
            "frame_recall": self.recall,
            "frame_f1": self.f1,
            "frame_iou": self.iou,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


def frame_level_metrics(
    pred_bouts: Sequence[Bout],
    ref_bouts: Sequence[Bout],
    n_frames: int,
    fps: float,
) -> FrameMetrics:
    pred = bouts_to_frame_labels(pred_bouts, n_frames, fps)
    ref = bouts_to_frame_labels(ref_bouts, n_frames, fps)

    tp = int(np.sum(pred & ref))
    fp = int(np.sum(pred & ~ref))
    fn = int(np.sum(~pred & ref))
    tn = int(np.sum(~pred & ~ref))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0

    return FrameMetrics(precision, recall, f1, iou, tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Bout-level metrics
# ---------------------------------------------------------------------------


def _bout_iou(a: Bout, b: Bout) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class BoutMetrics:
    matched: int
    false_positives: int
    missed: int
    mean_start_offset_s: float
    mean_end_offset_s: float
    median_iou: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "bout_matched": self.matched,
            "bout_fp": self.false_positives,
            "bout_missed": self.missed,
            "bout_start_offset_s": self.mean_start_offset_s,
            "bout_end_offset_s": self.mean_end_offset_s,
            "bout_median_iou": self.median_iou,
        }


def bout_level_metrics(
    pred_bouts: Sequence[Bout],
    ref_bouts: Sequence[Bout],
    iou_threshold: float = 0.3,
) -> BoutMetrics:
    """Greedy IoU bipartite matching between predicted and reference bouts."""
    if not pred_bouts and not ref_bouts:
        return BoutMetrics(0, 0, 0, 0.0, 0.0, 0.0)

    # All candidate pairs sorted by IoU descending
    pairs: List[Tuple[float, int, int]] = []
    for i, p in enumerate(pred_bouts):
        for j, r in enumerate(ref_bouts):
            iou = _bout_iou(p, r)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, i, j in pairs:
        if i in used_pred or j in used_ref:
            continue
        used_pred.add(i)
        used_ref.add(j)
        matches.append((i, j, iou))

    matched = len(matches)
    fp = len(pred_bouts) - matched
    missed = len(ref_bouts) - matched

    if matched > 0:
        start_offsets = [pred_bouts[i][0] - ref_bouts[j][0] for i, j, _ in matches]
        end_offsets = [pred_bouts[i][1] - ref_bouts[j][1] for i, j, _ in matches]
        ious = [iou for _, _, iou in matches]
        mean_start = float(np.mean(start_offsets))
        mean_end = float(np.mean(end_offsets))
        median_iou = float(np.median(ious))
    else:
        mean_start = mean_end = median_iou = 0.0

    return BoutMetrics(matched, fp, missed, mean_start, mean_end, median_iou)


# ---------------------------------------------------------------------------
# Loading predicted bouts from the per-chamber results CSV
# ---------------------------------------------------------------------------


def _extract_fly_index(label: str) -> Optional[int]:
    """Extract numeric fly index from `Fly 0 (Male)` style labels."""
    if not isinstance(label, str):
        return None
    for token in label.split():
        if token.isdigit():
            return int(token)
    return None


def load_predicted_courtship(
    results_csv: Path,
    male_only: bool = True,
) -> Tuple[List[Bout], str]:
    """Load Courtship bouts from a per-chamber results CSV.

    Returns (bouts, scope) where scope is one of:
    - 'male': filtered to the inferred-male fly
    - 'union': both flies merged (used when sex is unknown or male_only=False)
    """
    df = pd.read_csv(results_csv)
    df = df[df["Behavior"] == "Courtship"]
    if df.empty:
        return [], "empty"

    if male_only:
        male_df = df[df["Fly"].astype(str).str.contains(r"\(Male\)", regex=True)]
        if not male_df.empty:
            bouts = [(float(r["Start_Time (s)"]), float(r["End_Time (s)"]))
                     for _, r in male_df.iterrows()]
            return sorted(bouts), "male"

    # Union both flies
    raw = [(float(r["Start_Time (s)"]), float(r["End_Time (s)"]))
           for _, r in df.iterrows()]
    return _union_bouts(raw), "union"


def _union_bouts(bouts: Sequence[Bout]) -> List[Bout]:
    """Merge overlapping/abutting intervals."""
    if not bouts:
        return []
    sorted_bouts = sorted(bouts)
    merged: List[Bout] = [sorted_bouts[0]]
    for start, end in sorted_bouts[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


def evaluate_chamber(
    pred_bouts: Sequence[Bout],
    ref_bouts: Sequence[Bout],
    n_frames: int,
    fps: float,
    iou_threshold: float = 0.3,
) -> Dict[str, float]:
    frame = frame_level_metrics(pred_bouts, ref_bouts, n_frames, fps)
    bout = bout_level_metrics(pred_bouts, ref_bouts, iou_threshold)
    result = {
        **frame.as_dict(),
        **bout.as_dict(),
        "n_pred_bouts": len(pred_bouts),
        "n_ref_bouts": len(ref_bouts),
        "pred_total_s": sum(e - s for s, e in pred_bouts),
        "ref_total_s": sum(e - s for s, e in ref_bouts),
    }
    return result


def evaluate_video(
    ref_chambers: Sequence[ReferenceChamber],
    mapping: Dict[str, Optional[int]],
    results_dir: Path,
    fps: float,
    n_frames: int,
    male_only: bool = True,
    iou_threshold: float = 0.3,
) -> pd.DataFrame:
    """Run evaluation for every reference chamber that has a mapped detected index."""
    rows = []
    for chamber in ref_chambers:
        chamber_idx = mapping.get(chamber.descriptor)
        if chamber_idx is None:
            rows.append({
                "descriptor": chamber.descriptor,
                "chamber_idx": None,
                "scope": "unmapped",
            })
            continue

        results_csv = Path(results_dir) / f"chamber_{chamber_idx}_results.csv"
        if not results_csv.exists():
            rows.append({
                "descriptor": chamber.descriptor,
                "chamber_idx": chamber_idx,
                "scope": "no_results",
            })
            continue

        pred, scope = load_predicted_courtship(results_csv, male_only=male_only)
        metrics = evaluate_chamber(pred, chamber.bouts, n_frames, fps, iou_threshold)
        rows.append({
            "descriptor": chamber.descriptor,
            "chamber_idx": chamber_idx,
            "scope": scope,
            **metrics,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------------------


def plot_timeline(
    per_chamber_pred: Dict[str, List[Bout]],
    per_chamber_ref: Dict[str, List[Bout]],
    duration_s: float,
    title: str = "Predicted vs Reference Courtship",
):
    """Stacked horizontal bars: red = reference, blue = predicted, purple = overlap.

    Returns a matplotlib Figure. Caller is responsible for saving/displaying.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    descriptors = list(per_chamber_ref.keys())
    n_rows = max(1, len(descriptors))
    fig, ax = plt.subplots(figsize=(12, max(2.0, 0.6 * n_rows + 1.0)))

    bar_height = 0.35
    for i, desc in enumerate(descriptors):
        y = n_rows - i - 1
        ref_bouts = per_chamber_ref.get(desc, [])
        pred_bouts = per_chamber_pred.get(desc, [])

        for s, e in ref_bouts:
            ax.add_patch(Rectangle((s, y + 0.05), e - s, bar_height,
                                   facecolor="#d62728", edgecolor="none", alpha=0.8))
        for s, e in pred_bouts:
            ax.add_patch(Rectangle((s, y - bar_height - 0.05), e - s, bar_height,
                                   facecolor="#1f77b4", edgecolor="none", alpha=0.8))

    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.5, n_rows)
    ax.set_yticks([n_rows - i - 1 for i in range(n_rows)])
    ax.set_yticklabels(descriptors, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)

    # Legend
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="#d62728", label="Reference"))
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor="#1f77b4", label="Predicted"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig
