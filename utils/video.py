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

def detect_chambers(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Robust chamber detection designed for 4x3 grid (12 chambers).
    Uses morphological closing to bridge split chambers (tape) without merging columns.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    img_area = h * w
    
    # 1. Pre-process
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Threshold
    # Otsu is generally best for high contrast (bright chambers / black grid)
    thresh_val, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Smart Morphological Closing
    # We want to bridge the small vertical tape gap inside a chamber, 
    # BUT NOT bridge the large gap between columns.
    # Tape gap approx < 10px? Column gap approx > 20px?
    # Kernel width: 15px should bridge tape but stop at columns.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)) 
    bin_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Find Contours on the "healed" image
    cnts, _ = cv2.findContours(bin_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        
        # Filter size: A chamber is ~1/15th of the screen (12 chambers + margins)
        # Expected area roughly 5% - 10%
        # Use permissive bounds: 1% to 20%
        if area < img_area * 0.01 or area > img_area * 0.20:
            continue
            
        x, y, rw, rh = cv2.boundingRect(c)
        
        # Aspect Ratio Filter: Chambers are roughly squares or slightly tall rectangles
        # 0.5 to 1.5
        ar = float(rw) / rh
        if ar < 0.5 or ar > 1.8:
            continue
            
        candidates.append((x, y, rw, rh))
        
    # 5. Grid Enforcer (Target 12)
    # If we have > 12 candidates, pick the best 12 by size/regularity
    # If we have < 12, we might have missed some, but usually Otsu+Close is very good here.
    
    if len(candidates) > 12:
        # Sort by Area descending
        candidates.sort(key=lambda r: r[2]*r[3], reverse=True)
        # Take top 12 largest
        candidates = candidates[:12]
        
    # 6. Sort in Reading Order (Row 0: Col 0-2, Row 1: Col 0-2...)
    if candidates:
        # Determine approximate row height to bin Y coordinates
        # Median height of boxes
        heights = [r[3] for r in candidates]
        median_h = np.median(heights)
        
        # Sort by (Y_bin, X)
        # Bin Y by rounding to nearest row (row height approx median_h + gap)
        # Assuming gap is small, just dividing by (median_h * 0.7) separates rows effectively
        candidates.sort(key=lambda r: (int(r[1] / (median_h * 0.7)), r[0]))
        
    return candidates
