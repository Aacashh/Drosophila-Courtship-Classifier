from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d

# --- JAABA Feature Helpers ---

def parse_feature_name(tokens: Any) -> Dict[str, Any]:
    """Parse a JAABA window feature name token list into a dict.
    Expected tokens like: ['stat','mean','trans',<bitmask or string>,'radius',r,'offset',o,...]
    We accept flexible ordering and ignore unknown keys.
    """
    params: Dict[str, Any] = {}
    if isinstance(tokens, (list, tuple)):
        t = list(tokens)
    else:
        return params
    i = 0
    while i < len(t):
        k = t[i]
        v = t[i+1] if (i+1) < len(t) else None
        if isinstance(k, (str, bytes)):
            kk = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            params[kk] = v
        i += 2
    return params


def rolling_window_indices(n: int, radius: int, offset: int) -> List[Tuple[int, int]]:
    """Return (start, end) inclusive indices for each center t with window [t-r+off, t+r+off]."""
    idx: List[Tuple[int, int]] = []
    for t in range(n):
        s = max(0, (t - radius) + offset)
        e = min(n - 1, (t + radius) + offset)
        if e < s:
            s, e = e, s
        idx.append((s, e))
    return idx


def compute_window_feature(ts: np.ndarray, stat: str, radius: int, offset: int) -> np.ndarray:
    """Compute a simple window feature (mean/min/max/std/change) ignoring NaNs."""
    ts = np.asarray(ts, dtype=np.float32).copy()
    n = ts.size
    out = np.full(n, np.nan, dtype=np.float32)
    idxs = rolling_window_indices(n, radius, offset)
    if stat == 'mean':
        for t, (s, e) in enumerate(idxs):
            v = ts[s:e+1]
            out[t] = np.nanmean(v) if v.size else np.nan
    elif stat == 'min':
        for t, (s, e) in enumerate(idxs):
            v = ts[s:e+1]
            out[t] = np.nanmin(v) if v.size else np.nan
    elif stat == 'max':
        for t, (s, e) in enumerate(idxs):
            v = ts[s:e+1]
            out[t] = np.nanmax(v) if v.size else np.nan
    elif stat == 'std':
        for t, (s, e) in enumerate(idxs):
            v = ts[s:e+1]
            out[t] = np.nanstd(v) if v.size else np.nan
    elif stat == 'change':
        for t, (s, e) in enumerate(idxs):
            if e <= s:
                out[t] = 0.0
                continue
            mid = (s + e) // 2
            v0 = ts[s:mid+1]
            v1 = ts[mid+1:e+1]
            m0 = np.nanmean(v0) if v0.size else 0.0
            m1 = np.nanmean(v1) if v1.size else 0.0
            out[t] = m1 - m0
    else:
        out[:] = 0.0
    return out


def apply_transform(x: np.ndarray, trans: Any) -> np.ndarray:
    """Apply transform bitmask or string (none/abs/flip/relative)."""
    if isinstance(trans, (str, bytes)):
        tname = trans.decode() if isinstance(trans, (bytes, bytearray)) else trans
        tname = tname.lower()
        if tname == 'abs':
            return np.abs(x)
        if tname == 'flip':
            return -x
        if tname == 'relative':
            return _relative_transform(x)
        return x
    try:
        mask = int(trans)
    except Exception:
        return x
    y = x.copy()
    if mask & 2:
        y = np.abs(y)
    if mask & 4:
        y = -y
    if mask & 8:
        y = _relative_transform(y)
    return y


def _relative_transform(ts: np.ndarray) -> np.ndarray:
    x = ts[np.isfinite(ts)]
    if x.size == 0:
        return np.zeros_like(ts)
    prcBins = np.arange(0, 101, 2)
    relBins = np.percentile(x, prcBins)
    out = np.zeros_like(ts, dtype=np.float32)
    for i, v in enumerate(ts):
        if not np.isfinite(v):
            out[i] = 0.0
            continue
        idx = np.searchsorted(relBins, v, side='right')
        out[i] = float(idx) / float(len(relBins))
    return out


def compute_extended_window_feature(ts: np.ndarray, stat: str, radius: int, offset: int) -> np.ndarray:
    stat = stat.lower()
    if stat in ('mean','min','max','std','change'):
        return compute_window_feature(ts, stat, radius, offset)
    n = ts.size
    out = np.full(n, np.nan, dtype=np.float32)
    idxs = rolling_window_indices(n, radius, offset)
    if stat == 'diff_neighbor_mean':
        for t, (s, e) in enumerate(idxs):
            m = np.nanmean(ts[s:e+1]) if e >= s else np.nan
            out[t] = ts[t] - (m if np.isfinite(m) else 0.0)
    elif stat == 'diff_neighbor_min':
        for t, (s, e) in enumerate(idxs):
            m = np.nanmin(ts[s:e+1]) if e >= s else np.nan
            out[t] = ts[t] - (m if np.isfinite(m) else 0.0)
    elif stat == 'diff_neighbor_max':
        for t, (s, e) in enumerate(idxs):
            m = np.nanmax(ts[s:e+1]) if e >= s else np.nan
            out[t] = ts[t] - (m if np.isfinite(m) else 0.0)
    elif stat == 'zscore_neighbors':
        for t, (s, e) in enumerate(idxs):
            w = ts[s:e+1]
            m = np.nanmean(w) if w.size else 0.0
            sd = np.nanstd(w) if w.size else 1.0
            sd = sd if sd > 1e-6 else 1.0
            out[t] = (ts[t] - m) / sd
    else:
        out[:] = 0.0
    return out


def build_feature_matrix(perframe: Dict[str, List[np.ndarray]], feature_names: List[Any], fly_index: int) -> np.ndarray:
    """Given perframe data (tracks) and a JAABA-style feature_names token list, build a (nfeat x nframes) matrix for one fly."""
    rows: List[np.ndarray] = []
    for tokens in feature_names or []:
        params = parse_feature_name(tokens)
        stat = str(params.get('stat', 'mean'))
        radius = int(params.get('radius', 5) or 5)
        offset = int(params.get('offset', 0) or 0)
        pf_name = params.get('perframe') or params.get('feature') or params.get('src')
        if isinstance(pf_name, (list, tuple)):
            pf_name = pf_name[0]
        if isinstance(pf_name, (bytes, bytearray)):
            pf_name = pf_name.decode()

        synmap = {
            'angle': 'theta', 'absangle': 'theta', 'theta': 'theta',
            'speed': 'vel', 'velmag': 'vel', 'velocity': 'vel', 'dv': 'dv',
            'dist2wall': 'dist2wall', 'dcenter': 'dcenter',
            'dpartner': 'dpartner', 'dist_to_other': 'dpartner', 'distance2other': 'dpartner',
            'bearing': 'bearing_to_other', 'bearingtoother': 'bearing_to_other',
            'facing': 'facing_angle', 'facingangle': 'facing_angle', 'relangle': 'facing_angle',
            'wings': 'wing_angle', 'wing': 'wing_angle', 'wing_angle': 'wing_angle',
            'area': 'area', 'angspeed': 'angspeed', 'dtheta': 'angspeed'
        }
        if isinstance(pf_name, str):
            key = pf_name.lower().replace(' ', '').replace('_', '')
            for k, v in synmap.items():
                if key.startswith(k.replace('_','')):
                    pf_name = v
                    break

        if pf_name is None or pf_name not in perframe or fly_index >= len(perframe[pf_name]):
            candidates = ['vel', 'theta', 'dist2wall', 'dcenter', 'wing_angle', 'dpartner', 'area']
            found_name = next((c for c in candidates if c in perframe), None)
            if found_name and pf_name is None:
                pf_name = found_name

        if pf_name is None or pf_name not in perframe or fly_index >= len(perframe[pf_name]):
            ref_len = 0
            for v in perframe.values():
                if v and len(v) > fly_index:
                    ref_len = v[fly_index].size
                    break
            rows.append(np.zeros(ref_len if ref_len else 1, dtype=np.float32))
            continue

        ts = perframe[pf_name][fly_index]
        src = ts.copy()
        trans = params.get('trans') or params.get('trans_types') or 'none'
        src = apply_transform(src, trans)
        f = compute_extended_window_feature(src, stat, radius, offset)
        rows.append(f.astype(np.float32))

    if not rows:
        n = 1
        rows = [np.zeros(n, dtype=np.float32)]

    maxlen = max(r.size for r in rows)
    mat = np.zeros((len(rows), maxlen), dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(maxlen, r.size)
        mat[i, :n] = r[:n]
    return mat

# --- Heuristic Feature Helpers ---

def _smooth_positions(arr: np.ndarray, fps: float) -> np.ndarray:
    """Apply light Gaussian smoothing to a position array before computing velocity."""
    sigma = max(1.0, 0.05 * fps)  # ~50ms smoothing window
    # Handle NaN by interpolating, smoothing, then re-NaN-ing
    nans = np.isnan(arr)
    if np.all(nans):
        return arr.copy()
    if np.any(nans):
        clean = arr.copy()
        valid = ~nans
        indices = np.arange(len(arr))
        clean[nans] = np.interp(indices[nans], indices[valid], arr[valid])
        smoothed = gaussian_filter1d(clean, sigma=sigma)
        smoothed[nans] = np.nan
        return smoothed
    return gaussian_filter1d(arr, sigma=sigma)


def resolve_head_tail(tracks: Dict[str, List[np.ndarray]], n_flies: int = 2, fps: float = 30.0) -> Dict[str, List[np.ndarray]]:
    """
    Resolve the 180-degree ambiguity of elliptical orientation using velocity.
    Uses adaptive speed threshold and temporal hysteresis to prevent jitter.
    """
    n_frames = len(tracks['x'][0])
    if n_frames < 2:
        return tracks

    for i in range(n_flies):
        x = tracks['x'][i]
        y = tracks['y'][i]
        theta = tracks['theta'][i]

        # Smooth positions before computing velocity
        x_smooth = _smooth_positions(x, fps)
        y_smooth = _smooth_positions(y, fps)

        vx = np.gradient(x_smooth)
        vy = np.gradient(y_smooth)
        speed = np.hypot(vx, vy)
        mv_dir = np.arctan2(vy, vx)

        # Adaptive speed threshold: 20% of median speed (non-NaN frames)
        valid_speeds = speed[np.isfinite(speed) & (speed > 0)]
        if valid_speeds.size > 0:
            speed_threshold = max(0.3, 0.2 * np.median(valid_speeds))
        else:
            speed_threshold = 0.5

        new_theta = theta.copy()

        for t in range(n_frames):
            if np.isnan(theta[t]):
                continue

            # Normalize candidates
            t1 = theta[t] % (2 * np.pi)
            t2 = (theta[t] + np.pi) % (2 * np.pi)

            if speed[t] > speed_threshold:
                # Velocity-based choice
                md = mv_dir[t] % (2 * np.pi)

                d1 = abs(t1 - md)
                d1 = min(d1, 2 * np.pi - d1)
                d2 = abs(t2 - md)
                d2 = min(d2, 2 * np.pi - d2)

                vel_choice = t1 if d1 < d2 else t2

                # Hysteresis: prefer previous frame direction unless velocity strongly disagrees
                if t > 0 and not np.isnan(new_theta[t - 1]):
                    prev = new_theta[t - 1]
                    dp1 = abs(t1 - prev)
                    dp1 = min(dp1, 2 * np.pi - dp1)
                    dp2 = abs(t2 - prev)
                    dp2 = min(dp2, 2 * np.pi - dp2)
                    prev_choice = t1 if dp1 < dp2 else t2

                    if vel_choice != prev_choice:
                        # Only override continuity if velocity evidence is strong
                        if abs(d1 - d2) > 0.3:  # > ~17 degrees difference
                            new_theta[t] = vel_choice
                        else:
                            new_theta[t] = prev_choice
                    else:
                        new_theta[t] = vel_choice
                else:
                    new_theta[t] = vel_choice
            else:
                # Slow: prefer previous frame direction (temporal continuity)
                if t > 0 and not np.isnan(new_theta[t - 1]):
                    prev = new_theta[t - 1]
                    dp1 = abs(t1 - prev)
                    dp1 = min(dp1, 2 * np.pi - dp1)
                    dp2 = abs(t2 - prev)
                    dp2 = min(dp2, 2 * np.pi - dp2)
                    new_theta[t] = t1 if dp1 < dp2 else t2
                else:
                    # No previous frame — use tracker's pixel mass orientation
                    new_theta[t] = t1

        tracks['theta'][i] = new_theta

    return tracks


def compute_component_velocities(tracks: Dict[str, List[np.ndarray]], fly_idx: int, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward (v_par) and sideways (v_perp) velocities with temporal smoothing."""
    x = tracks['x'][fly_idx]
    y = tracks['y'][fly_idx]
    theta = tracks['theta'][fly_idx]

    # Smooth positions before computing gradients
    x_smooth = _smooth_positions(x, fps)
    y_smooth = _smooth_positions(y, fps)

    vx = np.gradient(x_smooth)
    vy = np.gradient(y_smooth)

    hx = np.cos(theta)
    hy = np.sin(theta)

    v_par = vx * hx + vy * hy
    v_perp = vx * (-hy) + vy * hx

    return v_par, v_perp


def compute_angular_velocity(tracks: Dict[str, List[np.ndarray]], fly_idx: int, fps: float = 30.0) -> np.ndarray:
    """Compute angular velocity (rad/s) with smoothing."""
    theta = tracks['theta'][fly_idx]
    dtheta = np.diff(theta, prepend=theta[0])
    # Wrap to [-pi, pi]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    ang_vel = dtheta * fps
    sigma = max(1.0, 0.05 * fps)
    nans = np.isnan(ang_vel)
    if np.any(nans) and not np.all(nans):
        clean = ang_vel.copy()
        valid = ~nans
        indices = np.arange(len(ang_vel))
        clean[nans] = np.interp(indices[nans], indices[valid], ang_vel[valid])
        smoothed = gaussian_filter1d(clean, sigma=sigma)
        smoothed[nans] = np.nan
        return smoothed
    elif np.all(nans):
        return ang_vel
    return gaussian_filter1d(ang_vel, sigma=sigma)


def compute_social_features(tracks: Dict[str, List[np.ndarray]], fly_idx: int, partner_idx: int, fps: float = 30.0) -> Dict[str, np.ndarray]:
    """Compute social features for fly_idx relative to partner_idx."""
    x1, y1, th1 = tracks['x'][fly_idx], tracks['y'][fly_idx], tracks['theta'][fly_idx]
    x2, y2, th2 = tracks['x'][partner_idx], tracks['y'][partner_idx], tracks['theta'][partner_idx]
    a2, b2 = tracks['a'][partner_idx], tracks['b'][partner_idx]

    # 1. Centroid to Centroid
    dx = x2 - x1
    dy = y2 - y1
    c2c = np.hypot(dx, dy)

    # 2. Facing Angle
    bearing = np.arctan2(dy, dx)
    facing_angle = bearing - th1
    facing_angle = (facing_angle + np.pi) % (2 * np.pi) - np.pi

    # 3. Nose to Ellipse (n2e) & Tail to Ellipse (t2e)
    a1 = tracks['a'][fly_idx]

    nose_x = x1 + a1 * np.cos(th1)
    nose_y = y1 + a1 * np.sin(th1)
    tail_x = x1 - a1 * np.cos(th1)
    tail_y = y1 - a1 * np.sin(th1)

    # Analytical ellipse radius approximation
    nose_dx = nose_x - x2
    nose_dy = nose_y - y2
    nose_phi = np.arctan2(nose_dy, nose_dx) - th2
    r_nose_dir = (a2 * b2) / np.sqrt((b2 * np.cos(nose_phi))**2 + (a2 * np.sin(nose_phi))**2 + 1e-10)
    dist_nose_center = np.hypot(nose_dx, nose_dy)
    n2e = np.maximum(0, dist_nose_center - r_nose_dir)

    tail_dx = tail_x - x2
    tail_dy = tail_y - y2
    tail_phi = np.arctan2(tail_dy, tail_dx) - th2
    r_tail_dir = (a2 * b2) / np.sqrt((b2 * np.cos(tail_phi))**2 + (a2 * np.sin(tail_phi))**2 + 1e-10)
    dist_tail_center = np.hypot(tail_dx, tail_dy)
    t2e = np.maximum(0, dist_tail_center - r_tail_dir)

    # Smoothed velocity magnitude
    x1_smooth = _smooth_positions(x1, fps)
    y1_smooth = _smooth_positions(y1, fps)
    vel = np.hypot(np.gradient(x1_smooth), np.gradient(y1_smooth))

    return {
        'c2c': c2c,
        'facing_angle': facing_angle,
        'n2e': n2e,
        't2e': t2e,
        'vel': vel
    }


def build_heuristic_features(tracks: Dict[str, List[np.ndarray]], fly_idx: int, fps: float = 30.0) -> Dict[str, np.ndarray]:
    """Build features for heuristic classification."""
    partner_idx = 1 - fly_idx

    feats = compute_social_features(tracks, fly_idx, partner_idx, fps)
    v_par, v_perp = compute_component_velocities(tracks, fly_idx, fps)
    feats['v_par'] = v_par
    feats['v_perp'] = v_perp
    feats['wing_angle'] = tracks['wing_angle'][fly_idx]
    feats['area'] = tracks['area'][fly_idx]
    feats['ang_vel'] = compute_angular_velocity(tracks, fly_idx, fps)

    return feats
