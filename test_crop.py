"""Test chamber detection and cropping."""
import time
import cv2
from pathlib import Path
from utils.video import detect_chambers, crop_chambers

VIDEO = Path("testing/V1.mp4")
OUT_DIR = Path("test_crop_output")

cap = cv2.VideoCapture(str(VIDEO))
ret, first_frame = cap.read()
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.release()
print(f"Video: {VIDEO}  ({total} frames, {fps:.1f} fps, {total/fps:.1f}s)")

chambers = detect_chambers(first_frame, padding=20, video_path=str(VIDEO))
print(f"Detected {len(chambers)} chambers\n")

t0 = time.time()
paths = crop_chambers(VIDEO, OUT_DIR, chambers)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f}s  ({total/elapsed:.0f} frames/sec)")
