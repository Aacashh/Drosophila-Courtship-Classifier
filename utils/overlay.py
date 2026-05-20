"""Overlay video writer — draws tracking ellipses and behavior labels on chamber videos."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any, Optional


def draw_chamber_overlay(
    frame: np.ndarray,
    bboxes: Sequence[Tuple[int, int, int, int]],
    labels: Optional[Sequence[Optional[str]]] = None,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    label_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw chamber bounding boxes + optional descriptors onto a copy of `frame`.

    Used by both the analyze tab and the verify tab so a single rendering
    is the source of truth.
    """
    out = frame.copy()
    for idx, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(out, f"#{idx}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
        if labels is not None and idx < len(labels) and labels[idx]:
            text = str(labels[idx])
            # Truncate long descriptors so they fit under the box
            if len(text) > 24:
                text = text[:21] + "..."
            cv2.putText(out, text, (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)
    return out


# Colors (BGR)
_COLOR_FLY0 = (255, 140, 0)   # blue-ish (male default)
_COLOR_FLY1 = (0, 140, 255)   # orange-ish (female default)
_COLORS = [_COLOR_FLY0, _COLOR_FLY1]

_BEHAVIOR_ABBREV = {
    'WingExt': 'Wing',
    'Following': 'Fol',
    'Circling': 'Cir',
    'Copulation': 'Cop',
    'Attempted_Copulation': 'AtCop',
    'Courtship': None,  # skip — it's the union, not a distinct behavior
}


def _build_behavior_masks(
    bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
    n_frames: int,
    n_flies: int,
) -> Dict[str, np.ndarray]:
    """Pre-compute boolean arrays (n_frames,) per behavior per fly.

    Returns {behavior_name: ndarray shape (n_flies, n_frames) dtype bool}.
    """
    masks = {}
    for beh, bouts_list in bouts_by_behavior.items():
        if beh == 'Courtship':
            continue  # skip union behavior
        arr = np.zeros((n_flies, n_frames), dtype=bool)
        for fly_idx, bouts in enumerate(bouts_list):
            if fly_idx >= n_flies:
                break
            for s, e in bouts:
                arr[fly_idx, max(0, s):min(n_frames, e + 1)] = True
        masks[beh] = arr
    return masks


def save_overlay_video(
    video_path: Path,
    output_path: Path,
    tracks: Dict[str, List[np.ndarray]],
    bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
    fly_stats: Dict[int, Dict[str, Any]],
    fps: float,
    male_fly: Optional[int] = None,
) -> None:
    """Read chamber video, draw tracking + behavior overlay, write output.

    Parameters
    ----------
    video_path : Path to cropped/stabilized chamber video.
    output_path : Path for annotated output video.
    tracks : Tracking result dict with keys x, y, theta, a, b, area, wing_angle.
    bouts_by_behavior : Classification results {behavior: [[bouts_fly0], [bouts_fly1]]}.
    fly_stats : Sex inference results {fly_idx: {'is_male': bool, ...}}.
    fps : Video frame rate.
    male_fly : Index of male fly (0 or 1). Auto-detected from fly_stats if None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return

    n_flies = len(tracks['x'])
    n_frames = len(tracks['x'][0])

    # Determine male fly index
    if male_fly is None:
        for fi, stats in fly_stats.items():
            if stats.get('is_male'):
                male_fly = fi
                break
        if male_fly is None:
            male_fly = 0

    # Fly labels
    fly_labels = []
    for fi in range(n_flies):
        if fi == male_fly:
            fly_labels.append('M')
        else:
            fly_labels.append('F')

    # Fly colors: male = blue, female = orange
    fly_colors = []
    for fi in range(n_flies):
        if fi == male_fly:
            fly_colors.append((255, 140, 0))   # blue-ish BGR
        else:
            fly_colors.append((0, 140, 255))    # orange BGR

    # Pre-compute behavior masks
    beh_masks = _build_behavior_masks(bouts_by_behavior, n_frames, n_flies)

    # Pre-extract track arrays for fast indexing
    xs = [tracks['x'][fi] for fi in range(n_flies)]
    ys = [tracks['y'][fi] for fi in range(n_flies)]
    thetas = [tracks['theta'][fi] for fi in range(n_flies)]
    semi_a = [tracks['a'][fi] for fi in range(n_flies)]  # semi-major / 2
    semi_b = [tracks['b'][fi] for fi in range(n_flies)]  # semi-minor / 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        # Draw flies
        for fi in range(n_flies):
            x = float(xs[fi][frame_idx])
            y = float(ys[fi][frame_idx])
            if np.isnan(x) or np.isnan(y):
                continue

            theta = float(thetas[fi][frame_idx])
            a = float(semi_a[fi][frame_idx])
            b = float(semi_b[fi][frame_idx])
            color = fly_colors[fi]

            # Draw ellipse
            angle_deg = np.degrees(theta)
            axes = (max(1, int(a)), max(1, int(b)))
            center = (int(round(x)), int(round(y)))
            cv2.ellipse(frame, center, axes, angle_deg, 0, 360, color, 1)

            # Draw heading arrow (8px)
            arrow_len = 8
            dx = arrow_len * np.cos(theta)
            dy = arrow_len * np.sin(theta)
            tip = (int(round(x + dx)), int(round(y + dy)))
            cv2.arrowedLine(frame, center, tip, color, 1, tipLength=0.4)

            # Identity label
            cv2.putText(frame, fly_labels[fi],
                        (center[0] + 6, center[1] - 6),
                        font, 0.3, color, 1, cv2.LINE_AA)

        # Active behaviors for male fly — text at top-left
        active = []
        for beh, mask in beh_masks.items():
            if mask[male_fly, frame_idx]:
                abbrev = _BEHAVIOR_ABBREV.get(beh, beh)
                if abbrev is not None:
                    active.append(abbrev)
        if active:
            label = ' | '.join(active)
            cv2.putText(frame, label, (4, 12), font, 0.35,
                        (0, 255, 0), 1, cv2.LINE_AA)

        # Timestamp at top-right
        t_sec = frame_idx / fps
        ts_text = f"{t_sec:.1f}s"
        (tw, _), _ = cv2.getTextSize(ts_text, font, 0.3, 1)
        cv2.putText(frame, ts_text, (width - tw - 4, 12), font, 0.3,
                    (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
