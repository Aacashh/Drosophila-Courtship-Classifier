import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TrackedFrame:
    x: List[float]
    y: List[float]
    theta: List[float]
    a: List[float]
    b: List[float]
    area: List[float]
    wing_angle: List[float]

def track_two_flies(video_path: Path, n_flies: int = 2) -> Dict[str, List[np.ndarray]]:
    """
    Track flies in a cropped chamber video.
    Returns a dictionary of per-frame arrays for features:
    x, y, theta, a, b, area, wing_angle, vel, etc.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Background Subtraction
    # History and threshold tuned for flies
    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20, detectShadows=False)
    
    # Data structures for tracks
    # separate lists for each fly
    tracks = {
        'x': [[] for _ in range(n_flies)],
        'y': [[] for _ in range(n_flies)],
        'theta': [[] for _ in range(n_flies)],
        'a': [[] for _ in range(n_flies)], # major axis
        'b': [[] for _ in range(n_flies)], # minor axis
        'area': [[] for _ in range(n_flies)],
        'wing_angle': [[] for _ in range(n_flies)]
    }
    
    prev_locs = None
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Segmentation
        fgmask = fgbg.apply(gray)
        
        # Clean up mask
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
        
        # Threshold to ensure binary
        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small noise
        min_area = (width * height) * 0.0005
        valid_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
        
        # Sort by area (largest first)
        valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:n_flies]
        
        current_detections = []
        
        for c in valid_cnts:
            area_val = float(cv2.contourArea(c))
            
            # Ellipse fitting
            if len(c) >= 5:
                (cx, cy), (ma, MA), angle = cv2.fitEllipse(c)
            else:
                # Fallback for small blobs
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    cx, cy = 0.0, 0.0
                MA, ma, angle = 1.0, 1.0, 0.0

            # Ensure MA >= ma
            if ma > MA:
                ma, MA = MA, ma
            
            # Wing Detection Logic
            # Create a mask for the fitted ellipse (the "body")
            body_mask = np.zeros_like(thresh)
            cv2.ellipse(body_mask, ((cx, cy), (ma, MA), angle), 255, -1)
            
            # Create a mask for the full contour (body + wings)
            contour_mask = np.zeros_like(thresh)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)
            
            # Subtract body from contour -> wings
            # We perform a bitwise XOR or simply subtract. 
            # Since body is an approximation, we intersect with the actual contour first.
            
            # Wings are parts of the contour NOT in the ellipse
            # But ellipse might be slightly smaller/larger.
            # Let's just look at the difference.
            # Pixels in contour but NOT in ellipse
            wings_mask = cv2.bitwise_and(contour_mask, cv2.bitwise_not(body_mask))
            
            wing_area = cv2.countNonZero(wings_mask)
            
            # Heuristic: wing angle is proportional to wing_area / body_area
            # Max wing extension is usually ~90 degrees (pi/2) per wing or total? 
            # JAABA expects something roughly in radians or degrees. FlyTracker outputs max angle.
            # Let's normalize. 
            # If wings are huge, angle is large.
            # Empirically, wing_area can be up to ~50% of body area during extension.
            ratio = wing_area / (area_val + 1e-6)
            # Simple scaling: if ratio is 0.3, angle might be ~60 degrees?
            # Let's assume max ratio ~0.5 corresponds to ~90 degrees.
            w_angle = min(90.0, (ratio / 0.5) * 90.0) 
            # We want radians often, but let's stick to degrees for now (FlyTracker uses degrees often, JAABA converts).
            # Wait, JAABA features often use radians. Tracker output usually consistent with input.
            # Let's output radians.
            w_angle_rad = np.radians(w_angle)
            
            current_detections.append({
                'x': cx, 'y': cy, 
                'theta': np.radians(angle), # Store in radians
                'a': MA/2, 'b': ma/2, 
                'area': area_val,
                'wing_angle': w_angle_rad
            })
            
        # Fill missing if fewer detections
        while len(current_detections) < n_flies:
            current_detections.append({
                'x': np.nan, 'y': np.nan, 'theta': np.nan, 
                'a': np.nan, 'b': np.nan, 'area': np.nan, 'wing_angle': np.nan
            })
            
        # Identity Assignment (Hungarian / Nearest Neighbor)
        assigned_dets = [None] * n_flies
        
        if prev_locs is None:
            # First frame: Sort by X coordinate (Left to Right)
            # Filter NaNs first for sorting
            valid_indices = [i for i, d in enumerate(current_detections) if not np.isnan(d['x'])]
            sorted_valid = sorted(valid_indices, key=lambda i: current_detections[i]['x'])
            
            # Assign valid ones
            for i, orig_idx in enumerate(sorted_valid):
                if i < n_flies:
                    assigned_dets[i] = current_detections[orig_idx]
            
            # Assign remaining (NaNs) to remaining slots
            used = set(sorted_valid[:n_flies])
            unused = [i for i in range(len(current_detections)) if i not in used]
            for i in range(n_flies):
                if assigned_dets[i] is None:
                    if unused:
                        assigned_dets[i] = current_detections[unused.pop(0)]
                    else:
                        assigned_dets[i] = {'x': np.nan, 'y': np.nan, 'theta': np.nan, 'a': np.nan, 'b': np.nan, 'area': np.nan, 'wing_angle': np.nan}

        else:
            # Nearest Neighbor
            # Cost matrix: distance
            # Using scipy linear_sum_assignment would be better but greedy is okay for 2 flies usually.
            # Let's use a simple distance matrix + greedy
            
            # Extract coords
            prev_pts = []
            curr_pts = []
            for i in range(n_flies):
                prev_pts.append(prev_locs[i]) # (x,y)
            
            # Current coords
            for d in current_detections:
                if np.isnan(d['x']):
                    curr_pts.append((np.inf, np.inf))
                else:
                    curr_pts.append((d['x'], d['y']))
            
            # Distance matrix
            dists = np.full((n_flies, n_flies), np.inf)
            for i in range(n_flies): # prev
                for j in range(n_flies): # curr
                    px, py = prev_pts[i]
                    cx, cy = curr_pts[j]
                    if np.isinf(px) or np.isinf(cx):
                        dists[i, j] = np.inf
                    else:
                        dists[i, j] = np.hypot(px-cx, py-cy)
            
            # Assign
            # If both infinite, assign based on index if possible or arbitrary
            from scipy.optimize import linear_sum_assignment
            # Replace inf with very large number for solver
            dists_safe = np.where(np.isinf(dists), 1e6, dists)
            row_ind, col_ind = linear_sum_assignment(dists_safe)
            
            for r, c in zip(row_ind, col_ind):
                # Check if distance is reasonable (e.g. not jumped across screen)
                # If infinite, it means we are matching a lost fly to a new detection or vice versa
                assigned_dets[r] = current_detections[c]
        
        # Update Tracks
        current_locs = []
        for i in range(n_flies):
            d = assigned_dets[i]
            for key in tracks:
                tracks[key][i].append(d.get(key, np.nan))
            
            # Update loc for next frame
            # If nan, keep previous loc? No, keep nan.
            # Actually, for tracking, if we lose a fly, we keep its last known pos for matching?
            # Or we mark it as lost.
            # Better to keep last known pos if lost for short time?
            # For simplicity, just use current.
            if not np.isnan(d['x']):
                current_locs.append((d['x'], d['y']))
            else:
                # If lost, maybe keep previous? 
                # Let's keep previous if available to prevent identity swap when it reappears?
                # Or use None/Inf.
                # If we use prev, we might latch onto the wrong one.
                # Let's stick to NaN/Inf for cost matrix.
                current_locs.append((np.inf, np.inf))
        
        prev_locs = current_locs
        frame_idx += 1

    cap.release()
    
    # Convert to numpy arrays
    final_tracks = {}
    for k in tracks:
        # list of arrays
        final_tracks[k] = [np.array(t, dtype=np.float32) for t in tracks[k]]
        
    # Compute derived features (velocity, etc.) needed for JAABA
    # We do this in features.py usually, but let's ensure we have the basics here.
    
    return final_tracks

