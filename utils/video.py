import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import subprocess
import shlex
import shutil

def stabilize_video(input_path: str, output_path: str, max_iter: int = 20, eps: float = 1e-3) -> bool:
    """Stabilize video by aligning all frames to the first frame using ECC.

    Optimizations over naive approach:
    - ECC computed on half-resolution images (4x faster per call)
    - Reduced max_iter (20 vs 50) and looser eps — sufficient for small chamber crops
    - Frames with negligible motion reuse previous warp for next N frames
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
        cap.release()
        return False

    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    # Downsampled reference for ECC (half resolution)
    scale = 0.5
    ref_small = cv2.resize(ref_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_small = np.eye(2, 3, dtype=np.float32)
    warp_full = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    out.write(ref)

    skip_counter = 0  # frames to skip ECC recomputation
    SKIP_FRAMES = 5   # reuse warp for this many frames when motion is negligible
    MOTION_THRESH = 0.5  # pixels — below this, skip ECC

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_counter > 0:
            # Reuse previous warp without recomputing ECC
            skip_counter -= 1
            aligned = cv2.warpAffine(frame, warp_full, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
            out.write(aligned)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # ECC on downsampled images
        gray_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            _, warp_small = cv2.findTransformECC(ref_small, gray_small, warp_small,
                                                  warp_mode, criteria, None, 1)
            # Scale translation back to full resolution
            warp_full = warp_small.copy()
            warp_full[0, 2] /= scale
            warp_full[1, 2] /= scale

            aligned = cv2.warpAffine(frame, warp_full, (w, h),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)

            # If motion is negligible, skip ECC for next N frames
            motion = np.hypot(warp_full[0, 2], warp_full[1, 2])
            if motion < MOTION_THRESH:
                skip_counter = SKIP_FRAMES
        except Exception:
            aligned = frame
        out.write(aligned)

    out.release()
    cap.release()
    return True


def crop_video(src: Path, dst: Path, roi: Tuple[int,int,int,int]) -> None:
    """Crop video to ROI (x, y, w, h). Uses ffmpeg if available, falls back to OpenCV."""
    x, y, w, h = roi

    # Try ffmpeg first (faster, better codec support)
    if shutil.which("ffmpeg"):
        try:
            import ffmpeg as ffmpeg_lib
            (
                ffmpeg_lib
                .input(str(src))
                .filter('crop', w, h, x, y)
                .output(str(dst), vcodec='libx264', preset='ultrafast', crf=23, pix_fmt='yuv420p')
                .overwrite_output()
                .run(quiet=True)
            )
            return
        except Exception:
            pass  # fall through to OpenCV

    # OpenCV fallback
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y:y+h, x:x+w]
        if cropped.shape[1] != w or cropped.shape[0] != h:
            cropped = cv2.resize(cropped, (w, h))
        out.write(cropped)
    cap.release()
    out.release()

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


def _snap_to_grid(candidates: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Given rough chamber detections, infer a regular grid and snap boxes to it.
    This ensures uniform box sizes that extend fully to the grid lines."""
    if len(candidates) < 2:
        return candidates

    cx = np.array([c[0] + c[2] / 2 for c in candidates])
    cy = np.array([c[1] + c[3] / 2 for c in candidates])
    median_h = np.median([c[3] for c in candidates])
    median_w = np.median([c[2] for c in candidates])

    # Cluster Y coords into rows
    sorted_indices_y = np.argsort(cy)
    sorted_cy = cy[sorted_indices_y]
    row_breaks = [0]
    for i in range(1, len(sorted_cy)):
        if sorted_cy[i] - sorted_cy[i - 1] > median_h * 0.4:
            row_breaks.append(i)
    row_centers = []
    for i in range(len(row_breaks)):
        start = row_breaks[i]
        end = row_breaks[i + 1] if i + 1 < len(row_breaks) else len(sorted_cy)
        row_centers.append(np.mean(sorted_cy[start:end]))

    # Cluster X coords into cols
    sorted_indices_x = np.argsort(cx)
    sorted_cx = cx[sorted_indices_x]
    col_breaks = [0]
    for i in range(1, len(sorted_cx)):
        if sorted_cx[i] - sorted_cx[i - 1] > median_w * 0.4:
            col_breaks.append(i)
    col_centers = []
    for i in range(len(col_breaks)):
        start = col_breaks[i]
        end = col_breaks[i + 1] if i + 1 < len(col_breaks) else len(sorted_cx)
        col_centers.append(np.mean(sorted_cx[start:end]))

    if len(row_centers) < 1 or len(col_centers) < 1:
        return candidates

    # Uniform cell size: 75th percentile (slightly larger to avoid clipping edges)
    widths = np.array([c[2] for c in candidates])
    heights = np.array([c[3] for c in candidates])
    cell_w = int(np.percentile(widths, 75))
    cell_h = int(np.percentile(heights, 75))

    snapped = []
    for ry in row_centers:
        for cx_val in col_centers:
            x = int(cx_val - cell_w / 2)
            y = int(ry - cell_h / 2)
            snapped.append((x, y, cell_w, cell_h))

    return snapped


def _grid_score(chambers: List[Tuple[int, int, int, int]]) -> float:
    """Score how grid-like a set of chamber detections is.
    Higher is better. Considers count, size consistency, and row/col regularity."""
    n = len(chambers)
    if n < 2:
        return float(n)

    areas = np.array([c[2] * c[3] for c in chambers])
    median_a = np.median(areas)
    if median_a == 0:
        return 0.0
    # Coefficient of variation of areas (lower = more consistent)
    cv = np.std(areas) / median_a
    size_score = max(0, 1.0 - cv)

    # Check row/col regularity via center-coordinate clustering
    cy = np.array([c[1] + c[3] / 2 for c in chambers])
    cx = np.array([c[0] + c[2] / 2 for c in chambers])
    median_h = np.median([c[3] for c in chambers])
    median_w = np.median([c[2] for c in chambers])

    # Count distinct rows and cols
    sorted_cy = np.sort(cy)
    n_rows = 1
    for i in range(1, len(sorted_cy)):
        if sorted_cy[i] - sorted_cy[i - 1] > median_h * 0.4:
            n_rows += 1
    sorted_cx = np.sort(cx)
    n_cols = 1
    for i in range(1, len(sorted_cx)):
        if sorted_cx[i] - sorted_cx[i - 1] > median_w * 0.4:
            n_cols += 1

    # Grid regularity: how close is n to n_rows * n_cols?
    expected = n_rows * n_cols
    grid_regularity = min(n, expected) / max(n, expected) if expected > 0 else 0

    return n * size_score * grid_regularity * (1.0 if n_rows > 1 or n_cols > 1 else 0.5)


def _extract_contour_candidates(binary: np.ndarray, img_area: int,
                                 min_frac: float = 0.002, max_frac: float = 0.12,
                                 ar_lo: float = 0.3, ar_hi: float = 3.0
                                 ) -> List[Tuple[int, int, int, int]]:
    """Find rectangular contour candidates from a binary image, filtered by area and aspect ratio."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * min_frac or area > img_area * max_frac:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = float(cw) / ch if ch > 0 else 0
        if ar < ar_lo or ar > ar_hi:
            continue
        candidates.append((x, y, cw, ch))
    return candidates


def _filter_size_outliers(candidates: List[Tuple[int, int, int, int]],
                          lo: float = 0.3, hi: float = 3.0
                          ) -> List[Tuple[int, int, int, int]]:
    """Reject candidates whose area deviates too far from the median."""
    if len(candidates) < 3:
        return candidates
    areas = np.array([c[2] * c[3] for c in candidates])
    median_area = np.median(areas)
    return [c for c, a in zip(candidates, areas) if lo * median_area < a < hi * median_area]


def _add_padding(candidates: List[Tuple[int, int, int, int]],
                 padding: int, img_w: int, img_h: int
                 ) -> List[Tuple[int, int, int, int]]:
    """Add padding to bounding boxes, clamped to image dimensions."""
    result = []
    for (x, y, w, h) in candidates:
        px = max(0, x - padding)
        py = max(0, y - padding)
        pw = min(img_w - px, w + 2 * padding)
        ph = min(img_h - py, h + 2 * padding)
        result.append((px, py, pw, ph))
    return result


def _strategy_direct_bright(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Strategy 1: Detect individual bright chambers directly on the full image.
    Works for white/bright chambers on a dark background (the most common arena type)."""
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If bright pixels dominate (>60%), chambers are likely the dark regions; invert.
    white_ratio = np.count_nonzero(binary) / img_area
    if white_ratio > 0.6:
        binary = cv2.bitwise_not(binary)

    # Light opening to cleanly separate adjacent chambers connected by thin noise bridges.
    # Use a small kernel so we break inter-chamber bridges but keep intra-chamber content.
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=2)

    # Small closing to fill tiny holes within a single chamber (tape marks, X-marks)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

    candidates = _extract_contour_candidates(binary, img_area)
    candidates = _filter_size_outliers(candidates)
    if len(candidates) >= 4:
        candidates = _snap_to_grid(candidates)
    return candidates


def _strategy_adaptive_threshold(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Strategy 2: Adaptive thresholding for arenas with uneven illumination."""
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Block size proportional to image but capped; must be odd
    block = max(31, min(w_img // 8, h_img // 8)) | 1
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, -5)

    white_ratio = np.count_nonzero(binary) / img_area
    if white_ratio > 0.6:
        binary = cv2.bitwise_not(binary)

    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=2)

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

    candidates = _extract_contour_candidates(binary, img_area)
    candidates = _filter_size_outliers(candidates)
    if len(candidates) >= 4:
        candidates = _snap_to_grid(candidates)
    return candidates


def _strategy_multi_threshold(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Strategy 3: Try multiple fixed thresholds and pick the one yielding the best grid."""
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    best_candidates = []
    best_score = 0.0

    # Sweep several thresholds in the bright range (chambers are white)
    for thresh_val in range(80, 220, 20):
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

        white_ratio = np.count_nonzero(binary) / img_area
        if white_ratio > 0.6:
            binary = cv2.bitwise_not(binary)
        if white_ratio < 0.01:
            continue

        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=2)

        candidates = _extract_contour_candidates(binary, img_area)
        candidates = _filter_size_outliers(candidates)

        if len(candidates) >= 4:
            snapped = _snap_to_grid(candidates)
            score = _grid_score(snapped)
            if score > best_score:
                best_score = score
                best_candidates = snapped

    return best_candidates


def _strategy_edge_based(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Strategy 4: Use Canny edges + dilation to find enclosed rectangular regions."""
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close small gaps in chamber boundaries
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k_dilate, iterations=2)

    # Flood fill from the edges to find enclosed regions
    # Invert so enclosed (non-edge) regions become white
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.002 or area > img_area * 0.12:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = float(cw) / ch if ch > 0 else 0
        if ar < 0.3 or ar > 3.0:
            continue
        # Rectangularity check: contour area vs bounding box area
        rect_ratio = area / (cw * ch) if cw * ch > 0 else 0
        if rect_ratio < 0.4:
            continue
        candidates.append((x, y, cw, ch))

    candidates = _filter_size_outliers(candidates)
    if len(candidates) >= 4:
        candidates = _snap_to_grid(candidates)
    return candidates


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
    """Adaptive chamber detection using multiple strategies with automatic selection.

    Tries four detection approaches and picks the result that yields the most
    grid-like arrangement of chambers:
      1. Direct bright-region detection (Otsu on full image)
      2. Adaptive thresholding (handles uneven illumination)
      3. Multi-threshold sweep (tries many fixed thresholds)
      4. Canny edge-based detection (finds enclosed rectangles)

    Parameters
    ----------
    frame : First frame of the video (used as fallback).
    padding : Extra pixels around each detected chamber.
    video_path : If provided, samples multiple frames for a stable reference.
    """
    stable = None
    if video_path is not None:
        stable = _get_stable_frame(video_path)
    work_frame = stable if stable is not None else frame

    if len(work_frame.shape) == 3:
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = work_frame

    h_img, w_img = gray.shape[:2]

    # Run all strategies and collect results
    strategies = [
        ("direct_bright", _strategy_direct_bright),
        ("adaptive_thresh", _strategy_adaptive_threshold),
        ("multi_threshold", _strategy_multi_threshold),
        ("edge_based", _strategy_edge_based),
    ]

    best_chambers = []
    best_score = -1.0

    for name, strategy_fn in strategies:
        try:
            chambers = strategy_fn(gray)
            if chambers:
                score = _grid_score(chambers)
                if score > best_score:
                    best_score = score
                    best_chambers = chambers
        except Exception:
            continue

    if not best_chambers:
        return []

    # Apply padding and clamp to image bounds
    best_chambers = _add_padding(best_chambers, padding, w_img, h_img)

    return _sort_reading_order(best_chambers)
