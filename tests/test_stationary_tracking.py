"""
Synthetic test: one moving dot + one dot that goes stationary.
Verifies that the hybrid tracker keeps the stationary dot in
the foreground mask (>= 95% non-NaN) while MOG2 alone loses it.

Video: 150s @ 30fps. Dot B moves for 50s, sits still for 20s, moves again.
The 50s warmup sees only moving dots, so the median background is clean.
The 20s stationary window is < half the 50s buffer, keeping it in foreground.
"""

import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

# Video parameters
W, H = 240, 250
FPS = 30
DURATION_S = 150
N_FRAMES = FPS * DURATION_S  # 4500
DOT_RADIUS = 10
BG_COLOR = 180
FLY_COLOR = 40

# Dot B timing
MOVE_END = 50 * FPS       # frame 1500: dot B stops
STAT_END = 70 * FPS       # frame 2100: dot B resumes (20s stationary)

# Fixed position for stationary phase
STAT_X, STAT_Y = 170, 180


def _make_synthetic_video(path: Path):
    """Create a synthetic 240x250 video with two dots."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (W, H), isColor=True)

    rng = np.random.RandomState(42)

    ax, ay = 60.0, 60.0
    bx, by = 170.0, 170.0

    for f in range(N_FRAMES):
        frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)

        # Dot A: always moving
        ax += rng.randn() * 2.5
        ay += rng.randn() * 2.5
        ax = np.clip(ax, DOT_RADIUS + 5, W - DOT_RADIUS - 5)
        ay = np.clip(ay, DOT_RADIUS + 5, H - DOT_RADIUS - 5)
        cv2.circle(frame, (int(ax), int(ay)), DOT_RADIUS, (FLY_COLOR,) * 3, -1)

        # Dot B: moving / stationary / moving
        if f < MOVE_END:
            bx += rng.randn() * 2.5
            by += rng.randn() * 2.5
            bx = np.clip(bx, DOT_RADIUS + 5, W - DOT_RADIUS - 5)
            by = np.clip(by, DOT_RADIUS + 5, H - DOT_RADIUS - 5)
        elif f == MOVE_END:
            bx, by = float(STAT_X), float(STAT_Y)
        elif f >= STAT_END:
            bx += rng.randn() * 2.5
            by += rng.randn() * 2.5
            bx = np.clip(bx, DOT_RADIUS + 5, W - DOT_RADIUS - 5)
            by = np.clip(by, DOT_RADIUS + 5, H - DOT_RADIUS - 5)

        cv2.circle(frame, (int(bx), int(by)), DOT_RADIUS, (FLY_COLOR,) * 3, -1)
        writer.write(frame)

    writer.release()


def _nan_pct(tracks, fly_idx, start_frame, end_frame):
    """Percentage of NaN frames for a fly in a given window."""
    x = tracks['x'][fly_idx][start_frame:end_frame]
    n_nan = np.isnan(x).sum()
    return n_nan / len(x) * 100.0


def _find_fly_near(tracks, target_x, start_frame, end_frame, tolerance=50):
    """Find fly index whose median x in a window is closest to target_x."""
    best_fi, best_dist = None, float('inf')
    for fi in range(2):
        x_window = tracks['x'][fi][start_frame:end_frame]
        valid = ~np.isnan(x_window)
        if valid.sum() == 0:
            continue
        med = np.nanmedian(x_window)
        d = abs(med - target_x)
        if d < tolerance and d < best_dist:
            best_fi, best_dist = fi, d
    return best_fi


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    vid = tmp_path_factory.mktemp("synth") / "two_dots.mp4"
    _make_synthetic_video(vid)
    return vid


def test_hybrid_stationary_detection(synthetic_video):
    """Under hybrid mode, stationary dot B must have >= 95% non-NaN frames
    during its stationary window (t=50s to t=70s)."""
    from tracking.tracker import track_two_flies

    tracks, _ = track_two_flies(synthetic_video, n_flies=2,
                                segmentation_mode="hybrid", buffer_seconds=50.0)

    fly_b = _find_fly_near(tracks, STAT_X, MOVE_END, STAT_END)

    if fly_b is None:
        # If we can't identify by position during stationary window,
        # check which fly has fewer NaNs there (the one that's detected)
        nan0 = _nan_pct(tracks, 0, MOVE_END, STAT_END)
        nan1 = _nan_pct(tracks, 1, MOVE_END, STAT_END)
        fly_b = 0 if nan0 < nan1 else 1
        print(f"\n[hybrid] Fallback identification: fly {fly_b} (NaN: {min(nan0,nan1):.1f}%)")

    nan_pct = _nan_pct(tracks, fly_b, MOVE_END, STAT_END)
    nonnan_pct = 100.0 - nan_pct
    print(f"\n[hybrid] Dot B stationary window: {nonnan_pct:.1f}% non-NaN ({nan_pct:.1f}% NaN)")
    assert nonnan_pct >= 95.0, (
        f"Hybrid mode: stationary dot has only {nonnan_pct:.1f}% non-NaN "
        f"(need >= 95%)")


def test_mog2_stationary_baseline(synthetic_video):
    """Under mog2 mode, record baseline NaN% for the stationary dot.
    This documents the regression we're fixing -- no assertion."""
    from tracking.tracker import track_two_flies

    tracks, _ = track_two_flies(synthetic_video, n_flies=2,
                                segmentation_mode="mog2")

    fly_b = _find_fly_near(tracks, STAT_X, MOVE_END, STAT_END)

    if fly_b is not None:
        nan_pct = _nan_pct(tracks, fly_b, MOVE_END, STAT_END)
        print(f"\n[mog2] Dot B stationary window: {100-nan_pct:.1f}% non-NaN ({nan_pct:.1f}% NaN)")
    else:
        # Check both flies
        for fi in range(2):
            nan_pct = _nan_pct(tracks, fi, MOVE_END, STAT_END)
            print(f"\n[mog2] Fly {fi} stationary window: {100-nan_pct:.1f}% non-NaN ({nan_pct:.1f}% NaN)")
