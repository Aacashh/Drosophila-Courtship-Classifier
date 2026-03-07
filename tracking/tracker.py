import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d

@dataclass
class TrackedFrame:
    x: List[float]
    y: List[float]
    theta: List[float]
    a: List[float]
    b: List[float]
    area: List[float]
    wing_angle: List[float]


def _nan_detection():
    """Return a NaN detection dict for missing flies."""
    return {'x': np.nan, 'y': np.nan, 'theta': np.nan,
            'a': np.nan, 'b': np.nan, 'area': np.nan, 'wing_angle': np.nan}


def interpolate_short_gaps(tracks: Dict[str, List[np.ndarray]], max_gap: int = 5) -> Dict[str, List[np.ndarray]]:
    """Interpolate short NaN gaps (up to max_gap frames) in tracking data."""
    interp_keys = ['x', 'y', 'a', 'b', 'area']
    angle_keys = ['theta']
    zero_keys = ['wing_angle']  # No data available for interpolated frames

    n_flies = len(tracks['x'])

    for fly_idx in range(n_flies):
        # Identify NaN gap runs
        nans = np.isnan(tracks['x'][fly_idx])
        if not np.any(nans):
            continue

        n = len(nans)
        # Find gap boundaries
        gap_starts = []
        gap_ends = []
        in_gap = False
        for t in range(n):
            if nans[t] and not in_gap:
                gap_starts.append(t)
                in_gap = True
            elif not nans[t] and in_gap:
                gap_ends.append(t - 1)
                in_gap = False
        if in_gap:
            gap_ends.append(n - 1)

        for gs, ge in zip(gap_starts, gap_ends):
            gap_len = ge - gs + 1
            if gap_len > max_gap:
                continue
            # Need valid frames on both sides
            if gs == 0 or ge == n - 1:
                continue
            if np.isnan(tracks['x'][fly_idx][gs - 1]) or np.isnan(tracks['x'][fly_idx][ge + 1]):
                continue

            # Linear interpolation for position/size features
            for key in interp_keys:
                arr = tracks[key][fly_idx]
                v_start = arr[gs - 1]
                v_end = arr[ge + 1]
                for t in range(gs, ge + 1):
                    frac = (t - gs + 1) / (gap_len + 1)
                    arr[t] = v_start + frac * (v_end - v_start)

            # Circular interpolation for theta
            for key in angle_keys:
                arr = tracks[key][fly_idx]
                th_start = arr[gs - 1]
                th_end = arr[ge + 1]
                # Unwrap for smooth interpolation
                diff = th_end - th_start
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                for t in range(gs, ge + 1):
                    frac = (t - gs + 1) / (gap_len + 1)
                    arr[t] = (th_start + frac * diff) % (2 * np.pi)

            # Zero for wing angle (no reliable data)
            for key in zero_keys:
                tracks[key][fly_idx][gs:ge + 1] = 0.0

    return tracks


def smooth_trajectories(tracks: Dict[str, List[np.ndarray]], sigma: float = 1.5) -> Dict[str, List[np.ndarray]]:
    """Apply light Gaussian smoothing to x, y trajectories to reduce jitter."""
    for key in ['x', 'y']:
        for fly_idx in range(len(tracks[key])):
            arr = tracks[key][fly_idx]
            nans = np.isnan(arr)
            if np.all(nans) or not np.any(~nans):
                continue
            if np.any(nans):
                clean = arr.copy()
                valid = ~nans
                indices = np.arange(len(arr))
                clean[nans] = np.interp(indices[nans], indices[valid], arr[valid])
                smoothed = gaussian_filter1d(clean, sigma=sigma)
                smoothed[nans] = np.nan
                tracks[key][fly_idx] = smoothed
            else:
                tracks[key][fly_idx] = gaussian_filter1d(arr, sigma=sigma)
    return tracks


def compute_overlap_flags(tracks: Dict[str, List[np.ndarray]], n_flies: int = 2) -> np.ndarray:
    """Flag frames where flies overlap (c2c < sum of semi-major axes)."""
    if n_flies < 2:
        return np.zeros(len(tracks['x'][0]), dtype=bool)

    x0, y0, a0 = tracks['x'][0], tracks['y'][0], tracks['a'][0]
    x1, y1, a1 = tracks['x'][1], tracks['y'][1], tracks['a'][1]

    c2c = np.hypot(x1 - x0, y1 - y0)
    body_threshold = a0 + a1  # sum of semi-major axes
    overlap = c2c < body_threshold

    # Also flag NaN frames
    overlap = overlap | np.isnan(x0) | np.isnan(x1)

    return overlap


def track_two_flies(video_path: Path, n_flies: int = 2) -> Dict[str, List[np.ndarray]]:
    """
    Track flies in a cropped chamber video.
    Returns a dictionary of per-frame arrays for features:
    x, y, theta, a, b, area, wing_angle, overlap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Max distance threshold for identity assignment (25% of chamber dimension)
    max_match_dist = 0.25 * min(width, height)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20, detectShadows=False)

    tracks = {
        'x': [[] for _ in range(n_flies)],
        'y': [[] for _ in range(n_flies)],
        'theta': [[] for _ in range(n_flies)],
        'a': [[] for _ in range(n_flies)],
        'b': [[] for _ in range(n_flies)],
        'area': [[] for _ in range(n_flies)],
        'wing_angle': [[] for _ in range(n_flies)]
    }

    prev_locs = None
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Segmentation
        fgmask = fgbg.apply(gray)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)

        _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = (width * height) * 0.0005
        valid_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
        valid_cnts = sorted(valid_cnts, key=cv2.contourArea, reverse=True)[:n_flies]

        current_detections = []

        for c in valid_cnts:
            area_val = float(cv2.contourArea(c))

            # Ellipse fitting
            if len(c) >= 5:
                (cx, cy), (ma, MA), angle = cv2.fitEllipse(c)
            else:
                M = cv2.moments(c)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    cx, cy = 0.0, 0.0
                MA, ma, angle = 1.0, 1.0, 0.0

            if ma > MA:
                ma, MA = MA, ma

            # --- Pixel Mass Orientation Check ---
            corrected_angle = angle

            roi_size = int(max(MA, ma) * 1.5)
            roi_size = max(roi_size, 20)
            x_min = max(0, int(cx - roi_size))
            y_min = max(0, int(cy - roi_size))
            x_max = min(width, int(cx + roi_size))
            y_max = min(height, int(cy + roi_size))

            if x_max > x_min and y_max > y_min:
                fly_roi = gray[y_min:y_max, x_min:x_max]

                roi_cx = cx - x_min
                roi_cy = cy - y_min

                M_rot = cv2.getRotationMatrix2D((roi_cx, roi_cy), angle, 1.0)

                h_roi, w_roi = fly_roi.shape
                cos_a = np.abs(M_rot[0, 0])
                sin_a = np.abs(M_rot[0, 1])
                nW = int((h_roi * sin_a) + (w_roi * cos_a))
                nH = int((h_roi * cos_a) + (w_roi * sin_a))

                M_rot[0, 2] += (nW / 2) - roi_cx
                M_rot[1, 2] += (nH / 2) - roi_cy

                rotated_roi = cv2.warpAffine(fly_roi, M_rot, (nW, nH))

                center_x = nW // 2
                left_half = rotated_roi[:, :center_x]
                right_half = rotated_roi[:, center_x:]

                if left_half.size > 0 and right_half.size > 0:
                    left_mass = np.sum(255 - left_half)
                    right_mass = np.sum(255 - right_half)

                    if right_mass > left_mass:
                        corrected_angle += 180

            corrected_angle = corrected_angle % 360.0
            theta_rad = np.radians(corrected_angle)

            # --- Geometric Wing Angle ---
            body_mask = np.zeros_like(thresh)
            cv2.ellipse(body_mask, ((cx, cy), (ma, MA), angle), 255, -1)

            contour_mask = np.zeros_like(thresh)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)

            wings_mask = cv2.bitwise_and(contour_mask, cv2.bitwise_not(body_mask))

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(wings_mask, connectivity=8)

            max_blob_area = 0
            wing_centroid = None

            for i in range(1, num_labels):
                a_i = stats[i, cv2.CC_STAT_AREA]
                # Validate wing blob: must be >5% and <50% of body area
                if a_i > max_blob_area and a_i > (area_val * 0.05) and a_i < (area_val * 0.5):
                    # Proximity check: wing must be close to body
                    wcx, wcy = centroids[i]
                    dist_to_body = np.hypot(wcx - cx, wcy - cy)
                    if dist_to_body < 2.0 * max(MA, ma):
                        max_blob_area = a_i
                        wing_centroid = centroids[i]

            w_angle_rad = 0.0

            if wing_centroid is not None:
                wx, wy = wing_centroid
                bx, by = cx, cy

                vw_x = wx - bx
                vw_y = wy - by

                vh_x = np.cos(theta_rad)
                vh_y = np.sin(theta_rad)

                norm_w = np.hypot(vw_x, vw_y)
                if norm_w > 1e-6:
                    dot = (vw_x * vh_x + vw_y * vh_y) / norm_w
                    dot = np.clip(dot, -1.0, 1.0)
                    w_angle_rad = np.arccos(dot)

            current_detections.append({
                'x': cx, 'y': cy,
                'theta': theta_rad,
                'a': MA / 2, 'b': ma / 2,
                'area': area_val,
                'wing_angle': w_angle_rad
            })

        # Fill missing if fewer detections
        while len(current_detections) < n_flies:
            current_detections.append(_nan_detection())

        # Identity Assignment (Hungarian / Nearest Neighbor)
        assigned_dets = [None] * n_flies

        if prev_locs is None:
            # First frame: Sort by X coordinate (Left to Right)
            valid_indices = [i for i, d in enumerate(current_detections) if not np.isnan(d['x'])]
            sorted_valid = sorted(valid_indices, key=lambda i: current_detections[i]['x'])

            for i, orig_idx in enumerate(sorted_valid):
                if i < n_flies:
                    assigned_dets[i] = current_detections[orig_idx]

            used = set(sorted_valid[:n_flies])
            unused = [i for i in range(len(current_detections)) if i not in used]
            for i in range(n_flies):
                if assigned_dets[i] is None:
                    if unused:
                        assigned_dets[i] = current_detections[unused.pop(0)]
                    else:
                        assigned_dets[i] = _nan_detection()
        else:
            prev_pts = list(prev_locs)

            curr_pts = []
            for d in current_detections:
                if np.isnan(d['x']):
                    curr_pts.append((np.inf, np.inf))
                else:
                    curr_pts.append((d['x'], d['y']))

            dists = np.full((n_flies, n_flies), np.inf)
            for i in range(n_flies):
                for j in range(n_flies):
                    px, py = prev_pts[i]
                    cx_j, cy_j = curr_pts[j]
                    if np.isinf(px) or np.isinf(cx_j):
                        dists[i, j] = np.inf
                    else:
                        dists[i, j] = np.hypot(px - cx_j, py - cy_j)

            from scipy.optimize import linear_sum_assignment
            dists_safe = np.where(np.isinf(dists), 1e6, dists)
            row_ind, col_ind = linear_sum_assignment(dists_safe)

            for r, c in zip(row_ind, col_ind):
                # Reject matches that are too far (prevent identity swaps)
                if dists[r, c] < max_match_dist or np.isinf(dists[r, c]):
                    # Accept match (inf means both lost — just propagate NaN)
                    if np.isinf(dists[r, c]):
                        assigned_dets[r] = _nan_detection()
                    else:
                        assigned_dets[r] = current_detections[c]
                else:
                    # Distance too large — mark as lost
                    assigned_dets[r] = _nan_detection()

            # Fill any remaining unassigned
            for i in range(n_flies):
                if assigned_dets[i] is None:
                    assigned_dets[i] = _nan_detection()

        # Update Tracks
        current_locs = []
        for i in range(n_flies):
            d = assigned_dets[i]
            for key in ['x', 'y', 'theta', 'a', 'b', 'area', 'wing_angle']:
                tracks[key][i].append(d.get(key, np.nan))

            # Keep last known position for matching (don't use inf for lost flies)
            if not np.isnan(d['x']):
                current_locs.append((d['x'], d['y']))
            else:
                # Retain previous valid position for future matching
                if prev_locs is not None and i < len(prev_locs) and not np.isinf(prev_locs[i][0]):
                    current_locs.append(prev_locs[i])
                else:
                    current_locs.append((np.inf, np.inf))

        prev_locs = current_locs
        frame_idx += 1

    cap.release()

    # Convert to numpy arrays
    final_tracks = {}
    for k in tracks:
        final_tracks[k] = [np.array(t, dtype=np.float32) for t in tracks[k]]

    # Post-processing: interpolate short NaN gaps
    final_tracks = interpolate_short_gaps(final_tracks, max_gap=5)

    # Post-processing: light trajectory smoothing
    final_tracks = smooth_trajectories(final_tracks, sigma=1.5)

    # Post-processing: compute overlap flags
    overlap = compute_overlap_flags(final_tracks, n_flies)
    final_tracks['overlap'] = [overlap]

    return final_tracks
