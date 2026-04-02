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


def _cluster_1d(values: np.ndarray, min_gap: float) -> List[float]:
    """Cluster sorted 1D values into groups separated by at least min_gap.
    Returns the mean of each cluster."""
    if len(values) == 0:
        return []
    sorted_vals = np.sort(values)
    clusters: List[List[float]] = [[float(sorted_vals[0])]]
    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] > min_gap:
            clusters.append([float(v)])
        else:
            clusters[-1].append(float(v))
    return [float(np.mean(c)) for c in clusters]


def _find_chamber_contours(gray: np.ndarray
                           ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Find bright rectangular chamber blobs using Otsu + fallback thresholds.

    Returns list of (contour, bounding_rect) pairs for chamber-sized regions.
    Tries Otsu first, then sweeps fixed thresholds if Otsu finds too few.
    """
    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    def _extract(binary: np.ndarray):
        # Ensure chambers are the white regions
        if np.count_nonzero(binary) / img_area > 0.6:
            binary = cv2.bitwise_not(binary)
        # Open to separate touching chambers, close to fill small holes
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=2)
        k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)

        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < img_area * 0.003 or area > img_area * 0.15:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            ar = cw / ch if ch > 0 else 0
            if ar < 0.3 or ar > 3.0:
                continue
            # Rectangularity: reject irregular blobs (border artifacts)
            rect_area = cw * ch
            if rect_area > 0 and area / rect_area < 0.5:
                continue
            results.append((c, (x, y, cw, ch)))
        return results

    # Try Otsu first
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates = _extract(binary)

    # Filter size outliers
    if len(candidates) >= 3:
        areas = np.array([r[2] * r[3] for _, r in candidates])
        med = np.median(areas)
        candidates = [(c, r) for (c, r), a in zip(candidates, areas)
                      if 0.4 * med < a < 2.5 * med]

    if len(candidates) >= 4:
        return candidates

    # Fallback: sweep fixed thresholds and pick the one with most chambers
    best = candidates
    for thresh in range(60, 220, 15):
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        cands = _extract(binary)
        if len(cands) >= 3:
            areas = np.array([r[2] * r[3] for _, r in cands])
            med = np.median(areas)
            cands = [(c, r) for (c, r), a in zip(cands, areas)
                     if 0.3 * med < a < 3.0 * med]
        if len(cands) > len(best):
            best = cands

    return best


def _detect_grid_angle(contours: List[np.ndarray]) -> float:
    """Detect grid rotation angle from the median orientation of chamber contours.

    Uses cv2.minAreaRect on each contour.  The angle is normalized to [-45, 45)
    degrees and converted to radians.
    """
    if len(contours) < 2:
        return 0.0

    angles_deg = []
    for c in contours:
        if len(c) < 5:
            continue
        rect = cv2.minAreaRect(c)
        w_r, h_r = rect[1]
        a = rect[2]  # degrees in [-90, 0)
        # minAreaRect angle convention: if width < height, add 90 to get
        # the angle of the long side relative to horizontal
        if w_r < h_r:
            a += 90.0
        # Normalize to [-45, 45)
        while a > 45.0:
            a -= 90.0
        while a < -45.0:
            a += 90.0
        angles_deg.append(a)

    if not angles_deg:
        return 0.0

    median_deg = float(np.median(angles_deg))
    # Ignore tiny angles (< 1°) — likely just noise
    if abs(median_deg) < 1.0:
        return 0.0
    return np.deg2rad(median_deg)


def _fit_grid(centers: np.ndarray, angle: float,
              median_w: float, median_h: float,
              img_w: int, img_h: int,
              gray: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int]]:
    """Fit a regular grid to detected chamber centers, accounting for rotation.

    1. Rotate centers to axis-aligned coordinate system
    2. Cluster x/y into columns/rows
    3. Compute cell size from center-to-center spacing
    4. Generate uniform grid cells
    5. Validate each cell has a detected center nearby (reject phantom cells)
    6. Rotate back and return axis-aligned bounding boxes
    """
    n = len(centers)
    if n < 2:
        return [(int(c[0] - median_w / 2), int(c[1] - median_h / 2),
                 int(median_w), int(median_h)) for c in centers]

    # Rotate centers to axis-aligned space around centroid
    centroid = np.mean(centers, axis=0)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot_fwd = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (centers - centroid) @ rot_fwd.T

    # Cluster into columns and rows
    col_centers = _cluster_1d(rotated[:, 0], median_w * 0.4)
    row_centers = _cluster_1d(rotated[:, 1], median_h * 0.4)

    if not col_centers or not row_centers:
        return [(int(c[0] - median_w / 2), int(c[1] - median_h / 2),
                 int(median_w), int(median_h)) for c in centers]

    # Regularize: make column/row centers evenly spaced
    col_centers = sorted(col_centers)
    row_centers = sorted(row_centers)

    if len(col_centers) >= 2:
        col_spacing = float(np.median(np.diff(col_centers)))
        col_start = col_centers[0]
        col_centers = [col_start + i * col_spacing
                       for i in range(len(col_centers))]
    else:
        col_spacing = median_w
    if len(row_centers) >= 2:
        row_spacing = float(np.median(np.diff(row_centers)))
        row_start = row_centers[0]
        row_centers = [row_start + i * row_spacing
                       for i in range(len(row_centers))]
    else:
        row_spacing = median_h

    # Cell size from spacing (prevents overlap) or from contour size
    cell_w = int(min(col_spacing - 4, median_w))
    cell_h = int(min(row_spacing - 4, median_h))

    # Safety floor
    cell_w = max(cell_w, int(median_w * 0.5))
    cell_h = max(cell_h, int(median_h * 0.5))

    # Precompute Otsu threshold for brightness validation
    otsu_thresh = 128.0
    if gray is not None:
        otsu_thresh = float(cv2.threshold(gray, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])

    # Generate grid and rotate back
    cos_b, sin_b = np.cos(angle), np.sin(angle)
    rot_back = np.array([[cos_b, -sin_b], [sin_b, cos_b]])

    chambers = []
    for ry in row_centers:
        for rx in col_centers:
            pt = np.array([rx, ry]) @ rot_back.T + centroid
            x = int(round(pt[0] - cell_w / 2))
            y = int(round(pt[1] - cell_h / 2))
            # Clamp to image bounds
            x = max(0, min(x, img_w - cell_w))
            y = max(0, min(y, img_h - cell_h))

            # Validate: check that the cell region is bright enough to be
            # a real chamber (rejects phantom cells at image borders)
            if gray is not None:
                cx1 = x + cell_w // 4
                cy1 = y + cell_h // 4
                cx2 = min(x + 3 * cell_w // 4, img_w)
                cy2 = min(y + 3 * cell_h // 4, img_h)
                region = gray[cy1:cy2, cx1:cx2]
                if region.size > 0:
                    bright_frac = np.count_nonzero(region > otsu_thresh) / region.size
                    if bright_frac < 0.2:
                        continue

            chambers.append((x, y, cell_w, cell_h))

    return chambers


def _add_padding(candidates: List[Tuple[int, int, int, int]],
                 padding: int, img_w: int, img_h: int
                 ) -> List[Tuple[int, int, int, int]]:
    """Add padding to bounding boxes, clamped to image dimensions and
    limited so adjacent boxes never overlap."""
    if not candidates or padding <= 0:
        return candidates

    # Max safe padding from center-to-center spacing
    centers_x = sorted(set(c[0] + c[2] / 2 for c in candidates))
    centers_y = sorted(set(c[1] + c[3] / 2 for c in candidates))
    median_w = np.median([c[2] for c in candidates])
    median_h = np.median([c[3] for c in candidates])

    max_pad_x = padding
    max_pad_y = padding
    if len(centers_x) >= 2:
        min_gap_x = min(centers_x[i + 1] - centers_x[i]
                        for i in range(len(centers_x) - 1))
        available_x = (min_gap_x - median_w) / 2
        max_pad_x = max(0, int(min(padding, available_x - 1)))
    if len(centers_y) >= 2:
        min_gap_y = min(centers_y[i + 1] - centers_y[i]
                        for i in range(len(centers_y) - 1))
        available_y = (min_gap_y - median_h) / 2
        max_pad_y = max(0, int(min(padding, available_y - 1)))

    result = []
    for (x, y, w, h) in candidates:
        px = max(0, x - max_pad_x)
        py = max(0, y - max_pad_y)
        pw = min(img_w - px, w + 2 * max_pad_x)
        ph = min(img_h - py, h + 2 * max_pad_y)
        result.append((px, py, pw, ph))
    return result


def _sort_reading_order(chambers: List[Tuple[int, int, int, int]],
                        angle: float = 0.0) -> List[Tuple[int, int, int, int]]:
    """Sort chambers in reading order (top-to-bottom, left-to-right).

    If the grid is rotated, transforms centers into the grid's own coordinate
    system before sorting so that rows/columns are identified correctly.
    """
    if not chambers:
        return chambers
    if abs(angle) < 0.001:
        # No rotation — simple row-binning sort
        heights = [r[3] for r in chambers]
        bin_size = max(np.median(heights) * 0.7, 1)
        return sorted(chambers, key=lambda r: (int(r[1] / bin_size), r[0]))

    # Rotate centers to axis-aligned space, sort there, map back
    centers = np.array([[c[0] + c[2] / 2, c[1] + c[3] / 2] for c in chambers])
    centroid = np.mean(centers, axis=0)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = (centers - centroid) @ rot.T

    median_h = np.median([c[3] for c in chambers])
    bin_size = max(median_h * 0.7, 1)

    indices = list(range(len(chambers)))
    indices.sort(key=lambda i: (int(rotated[i, 1] / bin_size), rotated[i, 0]))
    return [chambers[i] for i in indices]


def detect_chambers(frame: np.ndarray, padding: int = 15,
                    video_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
    """Detect chambers in a multi-chamber arena image.

    Pipeline:
      1. Get a stable reference frame (median of several frames)
      2. Find bright rectangular blobs (chamber candidates)
      3. Detect grid rotation from contour orientations
      4. Fit a regular grid to the detected centers
      5. Add padding, sort in reading order

    Handles different grid sizes, aspect ratios, and small rotations.

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

    # Step 1: Find chamber contours
    contour_results = _find_chamber_contours(gray)
    if len(contour_results) < 2:
        return []

    contours = [c for c, _ in contour_results]
    rects = [r for _, r in contour_results]
    centers = np.array([[r[0] + r[2] / 2, r[1] + r[3] / 2] for r in rects])
    median_w = float(np.median([r[2] for r in rects]))
    median_h = float(np.median([r[3] for r in rects]))

    # Step 2: Detect grid rotation
    angle = _detect_grid_angle(contours)

    # Step 3: Fit regular grid
    chambers = _fit_grid(centers, angle, median_w, median_h, w_img, h_img, gray)

    if not chambers:
        return []

    # Step 4: Padding and sort
    chambers = _add_padding(chambers, padding, w_img, h_img)
    return _sort_reading_order(chambers, angle)
