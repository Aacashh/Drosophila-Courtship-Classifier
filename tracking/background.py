# tracking/background.py
"""
Adaptive running-median background model.

The per-pixel median of a temporally-spaced frame buffer is robust to a fly
occluding any single pixel for many seconds. A stationary gravid female stays
in the foreground mask — unlike with MOG2, which decays her into background.

Ported idea (not code) from Ctrax's tracking approach.
"""

from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


class AdaptiveMedianBackground:
    """Per-pixel running median over a time-sampled buffer."""

    def __init__(
        self,
        shape: Tuple[int, int],
        buffer_seconds: float = 50.0,
        sample_hz: float = 1.0,
        min_dynamic_range: int = 15,
        open_kernel: int = 3,
    ):
        self._capacity = max(5, int(round(buffer_seconds * sample_hz)))
        self._stride_s = 1.0 / sample_hz
        self._buf: deque[np.ndarray] = deque(maxlen=self._capacity)
        self._last_sample_t: float = -np.inf
        self._cached_bg: Optional[np.ndarray] = None
        self._cache_dirty = True
        self._shape = shape
        self._min_dr = min_dynamic_range
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )

    # ---- buffer management -------------------------------------------------

    def update(self, gray: np.ndarray, frame_idx: int, fps: float) -> None:
        """Push frame into buffer if enough video-time has elapsed."""
        t = frame_idx / float(fps)
        if t - self._last_sample_t < self._stride_s:
            return
        self._buf.append(gray.copy())  # copy: caller may reuse the array
        self._last_sample_t = t
        self._cache_dirty = True

    def warmup(self, cap: cv2.VideoCapture, fps: float) -> None:
        """Seed the buffer from the first `buffer_seconds` of video.

        Accepts an already-open VideoCapture and seeks back to frame 0 when
        done, so the caller can continue reading from the beginning.
        """
        stride_frames = max(1, int(round(fps * self._stride_s)))
        idx = grabbed = 0
        while grabbed < self._capacity:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride_frames == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                self._buf.append(gray)
                grabbed += 1
            idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._cache_dirty = True
        # Prevent double-sampling in the first second of the tracking loop.
        if grabbed:
            self._last_sample_t = (grabbed - 1) * self._stride_s

    # ---- background / foreground ------------------------------------------

    def is_ready(self) -> bool:
        return len(self._buf) >= max(5, self._capacity // 4)

    def background(self) -> Optional[np.ndarray]:
        """Per-pixel median. Cached; recomputed only when the buffer changes."""
        if not self.is_ready():
            return None
        if self._cache_dirty or self._cached_bg is None:
            stack = np.stack(self._buf, axis=0)          # (N, H, W) uint8
            mid = stack.shape[0] // 2
            # partition is O(N) vs median's O(N log N); 2-3x faster on uint8.
            part = np.partition(stack, mid, axis=0)
            self._cached_bg = part[mid]
            self._cache_dirty = False
        return self._cached_bg

    def foreground(self, gray: np.ndarray) -> np.ndarray:
        """Binary FG mask via |gray - bg| + Otsu + morphological open."""
        bg = self.background()
        if bg is None:
            return np.zeros(self._shape, dtype=np.uint8)
        diff = cv2.absdiff(gray, bg)
        # Guard: Otsu returns garbage on uniform images (no fly present).
        if int(diff.max()) < self._min_dr:
            return np.zeros(self._shape, dtype=np.uint8)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return mask
