import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import subprocess
import shlex

def stabilize_video(input_path: str, output_path: str, max_iter: int = 50, eps: float = 1e-4) -> bool:
    """
    Stabilize video by aligning all frames to the first frame using ECC.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    ret, ref = cap.read()
    if not ret:
        return False

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (5,5), 0)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    out.write(ref)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        try:
            cc, warp = cv2.findTransformECC(ref_gray, gray, warp, warp_mode, criteria, None, 1)
            aligned = cv2.warpAffine(frame, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            aligned = frame
        out.write(aligned)

    out.release()
    cap.release()
    return True


def crop_video(src: Path, dst: Path, roi: Tuple[int,int,int,int]) -> None:
    """
    Crop video using ffmpeg-python.
    ROI: (x, y, w, h)
    """
    import ffmpeg
    x, y, w, h = roi
    (
        ffmpeg
        .input(str(src))
        .filter('crop', w, h, x, y)
        .output(str(dst), vcodec='libx264', preset='ultrafast', crf=23, pix_fmt='yuv420p')
        .overwrite_output()
        .run(quiet=True)
    )

def _get_stable_frame(video_path: str, n_samples: int = 5) -> Optional[np.ndarray]:
    """Sample multiple frames from the video and return the median to reduce
    transient occlusions (hands, shadows). Returns None if video can't be read."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None
    # Sample from the first quarter of the video (grid is static)
    indices = np.linspace(0, min(total - 1, total // 4), n_samples, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _find_grid_regions(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect one or more large grid panels in the frame.
    Returns bounding boxes (x, y, w, h) for each grid region found."""
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heavy closing to merge all chambers within a grid into one blob
    # Kernel proportional to image size so it works at any resolution
    kw = max(31, w_img // 15) | 1  # ensure odd
    kh = max(31, h_img // 15) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    merged = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Light opening to remove thin noise bridges between grids
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kw // 3 | 1, kh // 3 | 1))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, open_k, iterations=1)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grids = []
    for c in cnts:
        area = cv2.contourArea(c)
        # A grid panel should be at least 5% of the image
        if area < img_area * 0.05:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        # Reject very elongated blobs (not a grid)
        ar = float(rw) / rh if rh > 0 else 0
        if ar < 0.3 or ar > 3.0:
            continue
        grids.append((x, y, rw, rh))

    # Sort left-to-right
    grids.sort(key=lambda r: r[0])
    return grids


def _detect_chambers_in_region(gray: np.ndarray, region: Tuple[int, int, int, int],
                                padding: int = 15) -> List[Tuple[int, int, int, int]]:
    """Detect individual chambers within a grid region.
    Coordinates are returned in full-frame space."""
    rx, ry, rw, rh = region
    roi = gray[ry:ry+rh, rx:rx+rw]
    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small tape gaps within chambers
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    bin_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        # Chamber should be between 0.5% and 25% of the grid region
        if area < roi_area * 0.005 or area > roi_area * 0.25:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = float(cw) / ch if ch > 0 else 0
        if ar < 0.4 or ar > 2.5:
            continue
        candidates.append((x, y, cw, ch))

    if not candidates:
        return []

    # --- Grid structure validation ---
    # Filter by size consistency: reject outliers (hands, debris)
    areas = np.array([c[2] * c[3] for c in candidates])
    median_area = np.median(areas)
    # Keep chambers within 2x of median size
    candidates = [c for c, a in zip(candidates, areas) if 0.4 * median_area < a < 2.5 * median_area]

    if not candidates:
        return []

    # Apply padding and convert to full-frame coordinates
    full_h, full_w = gray.shape[:2]
    result = []
    for (x, y, cw, ch) in candidates:
        fx = max(0, rx + x - padding)
        fy = max(0, ry + y - padding)
        fw = min(full_w - fx, cw + 2 * padding)
        fh = min(full_h - fy, ch + 2 * padding)
        result.append((fx, fy, fw, fh))

    return result


def _sort_reading_order(chambers: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Sort chambers in reading order (top-to-bottom, left-to-right) using row binning."""
    if not chambers:
        return chambers
    heights = [r[3] for r in chambers]
    median_h = np.median(heights)
    bin_size = max(median_h * 0.7, 1)
    chambers.sort(key=lambda r: (int(r[1] / bin_size), r[0]))
    return chambers


def detect_chambers(frame: np.ndarray, padding: int = 15,
                    video_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
    """Adaptive chamber detection that handles single or multiple grid panels.

    Algorithm:
    1. Optionally use median of multiple video frames for occlusion robustness.
    2. Detect large grid regions (one or two panels).
    3. Within each grid, detect individual chambers.
    4. Validate by size consistency to reject transient objects (hands, shadows).
    5. Sort in reading order: grids left-to-right, chambers top-to-bottom within each grid.

    Parameters
    ----------
    frame : First frame of the video (used as fallback).
    padding : Extra pixels around each detected chamber.
    video_path : If provided, samples multiple frames for a stable reference.
    """
    # Use multi-frame median if video path available (robustness to hands/occlusions)
    stable = None
    if video_path is not None:
        stable = _get_stable_frame(video_path)
    work_frame = stable if stable is not None else frame

    if len(work_frame.shape) == 3:
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = work_frame

    # Step 1: Find grid regions
    grids = _find_grid_regions(gray)

    # Fallback: if no grid regions found, treat the whole frame as one grid
    if not grids:
        h_img, w_img = gray.shape[:2]
        grids = [(0, 0, w_img, h_img)]

    # Step 2: Detect chambers within each grid
    all_chambers = []
    for grid in grids:
        chambers = _detect_chambers_in_region(gray, grid, padding)
        chambers = _sort_reading_order(chambers)
        all_chambers.extend(chambers)

    return all_chambers
