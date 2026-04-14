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


def _get_ffmpeg_exe() -> Optional[str]:
    """Return path to ffmpeg binary, checking system PATH then imageio_ffmpeg."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def crop_chambers(src: Path, dst_dir: Path,
                  rois: List[Tuple[int, int, int, int]]) -> List[Path]:
    """Crop all chambers from a video.

    Uses ffmpeg subprocess per chamber (hardware-accelerated decode, fast
    h264 encode) if available. Falls back to single-pass OpenCV.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg:
        # ffmpeg path: one subprocess per chamber, runs in parallel
        procs = []
        for i, (x, y, w, h) in enumerate(rois):
            p = dst_dir / f"chamber_{i+1}.mp4"
            paths.append(p)
            cmd = [
                ffmpeg, "-y",
                "-i", str(src),
                "-filter:v", f"crop={w}:{h}:{x}:{y}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an",
                str(p),
            ]
            procs.append(subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            print(f"  Started ffmpeg for chamber {i+1}/{len(rois)}")

        # Wait for all to finish
        for i, proc in enumerate(procs):
            proc.wait()
            print(f"  Chamber {i+1} done")

        return paths

    # OpenCV fallback: single-pass decode, slice all ROIs
    cap = cv2.VideoCapture(str(src))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []
    for i, (_, _, w, h) in enumerate(rois):
        p = dst_dir / f"chamber_{i+1}.mp4"
        writers.append(cv2.VideoWriter(str(p), fourcc, fps, (w, h)))
        paths.append(p)

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for j, (x, y, w, h) in enumerate(rois):
            writers[j].write(frame[y:y+h, x:x+w])
        n += 1
        if n % 1000 == 0:
            print(f"  {n}/{total} frames")

    cap.release()
    for wr in writers:
        wr.release()

    return paths


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


def find_best_detection_frame(video_path: str,
                              start_sec: float = 90,
                              end_sec: float = 300,
                              step_sec: float = 5) -> Optional[np.ndarray]:
    """Find the best frame for chamber detection by sampling after divider removal.

    Scans frames from start_sec to end_sec, picks the one with the most
    chamber-sized bright blobs (dividers already removed, arena fully visible).
    Returns a median-composite of nearby frames for noise reduction.

    Parameters
    ----------
    video_path : Path to the video file.
    start_sec  : Earliest time to sample (well after dividers removed, ~90s).
    end_sec    : Latest time to sample.
    step_sec   : Interval between samples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps
    if total < 1:
        cap.release()
        return None

    start = min(start_sec, duration * 0.15)
    end = min(end_sec, duration * 0.8)

    best_count = 0
    best_t = start

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    t = start
    while t < end:
        idx = int(t * fps)
        if idx >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            t += step_sec
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_area = h * w

        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        eroded = cv2.erode(binary, k, iterations=2)
        recovered = cv2.dilate(eroded, k, iterations=2)
        cnts, _ = cv2.findContours(recovered, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        count = sum(1 for c in cnts
                    if img_area * 0.002 < cv2.contourArea(c) < img_area * 0.08)
        if count > best_count:
            best_count = count
            best_t = t
        t += step_sec

    # Median of 7 frames around the best time for noise reduction
    nearby = np.linspace(max(0, best_t * fps - fps * 3),
                         min(total - 1, best_t * fps + fps * 3),
                         7, dtype=int)
    frames = []
    for idx in nearby:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, f = cap.read()
        if ret:
            frames.append(f.astype(np.float32))
    cap.release()

    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _detect_blobs(binary: np.ndarray, img_area: int,
                  kernel_size: int, iterations: int
                  ) -> List[Tuple[int, int, int, int, float]]:
    """Run erode→dilate on a binary image and return chamber-shaped blobs.

    Returns list of (x, y, w, h, area) for blobs passing size/shape filters.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (kernel_size, kernel_size))
    eroded = cv2.erode(binary, k, iterations=iterations)
    recovered = cv2.dilate(eroded, k, iterations=iterations)

    cnts, _ = cv2.findContours(recovered, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.002 or area > img_area * 0.08:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / bh if bh > 0 else 0
        if ar < 0.5 or ar > 2.0:
            continue
        rect_fill = area / (bw * bh) if bw * bh > 0 else 0
        if rect_fill < 0.45:
            continue
        candidates.append((x, y, bw, bh, area))

    # Remove area outliers — all chambers should be nearly the same size
    if len(candidates) >= 4:
        areas = np.array([c[4] for c in candidates])
        med = np.median(areas)
        candidates = [c for c in candidates if 0.4 * med < c[4] < 2.0 * med]

    return candidates


def detect_chambers(frame: np.ndarray, padding: int = 20,
                    video_path: Optional[str] = None) -> List[Tuple[int, int, int, int]]:
    """Detect chambers in a multi-chamber arena image.

    Pipeline:
      1. Get the best detection frame (after divider removal, ~60s+)
      2. Otsu threshold to find bright regions
      3. Adaptive erosion: try multiple kernel sizes, pick the one that
         finds the most chamber-sized blobs (handles varying divider thickness)
      4. Filter by area / aspect ratio / rectangularity
      5. Remove area outliers, add padding, sort in reading order

    Parameters
    ----------
    frame : First frame of the video (used as fallback).
    padding : Extra pixels around each detected chamber.
    video_path : If provided, finds the best detection frame automatically.
    """
    # Use best detection frame if video is available, else fall back to given frame
    work_frame = frame
    if video_path is not None:
        best = find_best_detection_frame(video_path)
        if best is not None:
            work_frame = best

    if len(work_frame.shape) == 3:
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = work_frame

    h_img, w_img = gray.shape[:2]
    img_area = h_img * w_img

    # Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive erosion: try multiple kernel/iteration combos,
    # pick the one that finds the most chamber-sized blobs.
    # Stronger erosion separates touching chambers but may erase dim ones.
    best_candidates = []
    for ks, iters in [(5, 2), (7, 2), (9, 2), (7, 3), (9, 3), (11, 3)]:
        cands = _detect_blobs(binary, img_area, ks, iters)
        if len(cands) > len(best_candidates):
            best_candidates = cands

    if not best_candidates:
        return []

    # Add padding, clamped to image bounds
    chambers = []
    for x, y, bw, bh, _ in best_candidates:
        px = max(0, x - padding)
        py = max(0, y - padding)
        pw = min(w_img - px, bw + 2 * padding)
        ph = min(h_img - py, bh + 2 * padding)
        chambers.append((px, py, pw, ph))

    # Sort in reading order (top-to-bottom, left-to-right)
    if chambers:
        med_h = np.median([c[3] for c in chambers])
        chambers.sort(key=lambda r: (int(r[1] / (med_h * 0.5)), r[0]))

    return chambers
