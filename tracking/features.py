from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

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
        # difference between mean in the last subwindow and first subwindow: use halves
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
        # Unsupported stat -> zeros
        out[:] = 0.0
    return out


def apply_transform(x: np.ndarray, trans: Any) -> np.ndarray:
    """Apply transform bitmask or string (none/abs/flip/relative)."""
    # JAABA uses a bitmask; we map basic transforms heuristically
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
    # If numeric bitmask, apply abs (2) and flip (4); relative (8)
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
    # Approximate JAABA relative transform using percentiles over finite samples
    x = ts[np.isfinite(ts)]
    if x.size == 0:
        return np.zeros_like(ts)
    prcBins = np.arange(0, 101, 2)
    relBins = np.percentile(x, prcBins)
    # Map each value to its percentile bin center index normalized [0,1]
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
        # Unsupported stat -> zeros
        out[:] = 0.0
    return out


def build_feature_matrix(perframe: Dict[str, List[np.ndarray]], feature_names: List[Any], fly_index: int) -> np.ndarray:
    """Given perframe data (tracks) and a JAABA-style feature_names token list, build a (nfeat x nframes) matrix for one fly.
    This implements a useful subset: stats in {mean,min,max,std,change}, transforms {none,abs,flip}. Others map to zeros.
    """
    rows: List[np.ndarray] = []
    # feature_names is often a list of token lists per window feature
    for tokens in feature_names or []:
        params = parse_feature_name(tokens)
        stat = str(params.get('stat', 'mean'))
        radius = int(params.get('radius', 5) or 5)
        offset = int(params.get('offset', 0) or 0)
        # Parse perframe source; in many cases, tokens include something like 'perframe','<name>'
        # Fall back to commonly used names if unspecified
        pf_name = params.get('perframe') or params.get('feature') or params.get('src')
        if isinstance(pf_name, (list, tuple)):
            pf_name = pf_name[0]
        if isinstance(pf_name, (bytes, bytearray)):
            pf_name = pf_name.decode()
        
        # Normalize and map synonyms
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
        
        # Check availability
        # perframe values are lists of arrays [fly0, fly1, ...]
        if pf_name is None or pf_name not in perframe or fly_index >= len(perframe[pf_name]):
            # Fallback? or zeros
            # Try candidates if name is ambiguous
            candidates = ['vel', 'theta', 'dist2wall', 'dcenter', 'wing_angle', 'dpartner', 'area']
            found_name = next((c for c in candidates if c in perframe), None)
            if found_name and pf_name is None: # Only fallback if we had NO name
                pf_name = found_name
            
        if pf_name is None or pf_name not in perframe or fly_index >= len(perframe[pf_name]):
            # Append zeros with same length as others if possible
            # But we don't know length yet if it's the first row.
            # Assume standard length if we have any data.
            ref_len = 0
            for v in perframe.values():
                if v and len(v) > fly_index:
                    ref_len = v[fly_index].size
                    break
            rows.append(np.zeros(ref_len if ref_len else 1, dtype=np.float32))
            continue
            
        ts = perframe[pf_name][fly_index]
        
        # Apply transform to source series before computing the window stat
        src = ts.copy()
        trans = params.get('trans') or params.get('trans_types') or 'none'
        src = apply_transform(src, trans)
        
        # Extended stats first, fallback to base
        f = compute_extended_window_feature(src, stat, radius, offset)
        rows.append(f.astype(np.float32))
        
    # Normalize shapes; pad with zeros if empty
    if not rows:
        n = 1
        rows = [np.zeros(n, dtype=np.float32)]
    
    # Align frames
    maxlen = max(r.size for r in rows)
    mat = np.zeros((len(rows), maxlen), dtype=np.float32)
    for i, r in enumerate(rows):
        n = min(maxlen, r.size)
        mat[i, :n] = r[:n]
    return mat

# --- Heuristic Feature Helpers ---

def resolve_head_tail(tracks: Dict[str, List[np.ndarray]], n_flies: int = 2) -> Dict[str, List[np.ndarray]]:
    """
    Resolve the 180-degree ambiguity of elliptical orientation using velocity.
    Flies mostly walk forward.
    Updates 'theta' in tracks to be the heading direction (0-2pi).
    Adds 'vx', 'vy' if not present.
    """
    # Check if we have enough frames
    n_frames = len(tracks['x'][0])
    if n_frames < 2:
        return tracks

    for i in range(n_flies):
        x = tracks['x'][i]
        y = tracks['y'][i]
        theta = tracks['theta'][i] # from fitEllipse, usually 0-pi or similar
        
        # Calculate velocity
        vx = np.gradient(x)
        vy = np.gradient(y)
        speed = np.hypot(vx, vy)
        mv_dir = np.arctan2(vy, vx) # -pi to pi
        
        # Correct theta
        # fitEllipse angle is usually in degrees 0-180 or 0-360?
        # Our tracker stores it in radians.
        # We want theta to point towards the head.
        # If speed is significant, align theta with movement.
        
        new_theta = theta.copy()
        
        # Smooth speed/dir?
        
        for t in range(n_frames):
            if np.isnan(theta[t]):
                continue
                
            # Check alignment with velocity
            if speed[t] > 0.5: # 0.5 px/frame threshold
                # Ellipse angle is bidirectional: theta or theta + pi
                # Find which is closer to mv_dir
                
                # Normalize to 0-2pi
                t1 = theta[t] % (2*np.pi)
                t2 = (theta[t] + np.pi) % (2*np.pi)
                
                md = mv_dir[t] % (2*np.pi)
                
                # Distances
                d1 = abs(t1 - md)
                d1 = min(d1, 2*np.pi - d1)
                
                d2 = abs(t2 - md)
                d2 = min(d2, 2*np.pi - d2)
                
                if d1 < d2:
                    new_theta[t] = t1
                else:
                    new_theta[t] = t2
            else:
                # If slow, prioritize the Pixel Mass Orientation from tracker.
                # We assume tracker.py has already set theta correctly based on body intensity.
                new_theta[t] = theta[t]
                        
        tracks['theta'][i] = new_theta
        
    return tracks

def compute_component_velocities(tracks: Dict[str, List[np.ndarray]], fly_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute forward (v_par) and sideways (v_perp) velocities.
    """
    x = tracks['x'][fly_idx]
    y = tracks['y'][fly_idx]
    theta = tracks['theta'][fly_idx]
    
    # Gradient for velocity
    vx = np.gradient(x)
    vy = np.gradient(y)
    
    # Heading vector
    hx = np.cos(theta)
    hy = np.sin(theta)
    
    # Forward velocity: projection onto heading
    v_par = vx * hx + vy * hy
    
    # Sideways velocity: projection onto orthogonal vector (-sin, cos)
    # This gives positive value for movement to the left of the fly
    v_perp = vx * (-hy) + vy * hx
    
    return v_par, v_perp

def compute_social_features(tracks: Dict[str, List[np.ndarray]], fly_idx: int, partner_idx: int) -> Dict[str, np.ndarray]:
    """
    Compute social features for fly_idx relative to partner_idx.
    Returns dict of arrays: c2c, n2e, t2e, facing_angle, etc.
    """
    x1, y1, th1 = tracks['x'][fly_idx], tracks['y'][fly_idx], tracks['theta'][fly_idx]
    x2, y2, th2 = tracks['x'][partner_idx], tracks['y'][partner_idx], tracks['theta'][partner_idx]
    a2, b2 = tracks['a'][partner_idx], tracks['b'][partner_idx] # Major/Minor semi-axes of partner
    
    # 1. Centroid to Centroid
    dx = x2 - x1
    dy = y2 - y1
    c2c = np.hypot(dx, dy)
    
    # 2. Facing Angle (angle between fly1 heading and vector to fly2)
    # vector to fly2
    bearing = np.arctan2(dy, dx)
    facing_angle = bearing - th1
    # wrap to -pi, pi
    facing_angle = (facing_angle + np.pi) % (2*np.pi) - np.pi
    
    # 3. Nose to Ellipse (n2e) & Tail to Ellipse (t2e)
    # Fly1 Nose/Tail positions
    # Head is at theta
    # Tail is at theta + pi
    # Length is 2*a1 ? No, we need a1 (semi-major)
    a1 = tracks['a'][fly_idx]
    
    nose_x = x1 + a1 * np.cos(th1)
    nose_y = y1 + a1 * np.sin(th1)
    
    tail_x = x1 - a1 * np.cos(th1)
    tail_y = y1 - a1 * np.sin(th1)
    
    # Distance to Ellipse 2
    # Ellipse 2 defined by x2, y2, a2, b2, th2.
    # Analytical distance to ellipse is hard.
    # Approx: Distance to centroid minus radius in that direction?
    # Or closest point on ellipse?
    # Heuristic from repo: "Nearest distance from nose of fly1 to any point on the ellipse fitted to fly2."
    # We can sample points on ellipse 2 and find min dist.
    
    n_frames = len(x1)
    n2e = np.zeros(n_frames)
    t2e = np.zeros(n_frames)
    
    # Vectorize sampling? 
    # Ellipse points: P(t) = Center + R(th2) * [a2*cos(phi), b2*sin(phi)]
    # We can use just 16 points around the ellipse for approx
    phi = np.linspace(0, 2*np.pi, 16)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # We have to loop frames or broadcast heavily.
    # Let's loop frames for clarity/safety, or block process.
    # To be fast in python, maybe just use centroid distance minus approx radius?
    # Radius at angle alpha (relative to ellipse axis) is:
    # r(alpha) = a*b / sqrt((b*cos)^2 + (a*sin)^2)
    # This is much faster.
    
    # Angle of nose relative to ellipse 2 center
    nose_dx = nose_x - x2
    nose_dy = nose_y - y2
    nose_phi = np.arctan2(nose_dy, nose_dx) - th2 # Angle relative to ellipse axis
    
    # Radius of ellipse 2 in direction of nose
    r_nose_dir = (a2 * b2) / np.sqrt((b2 * np.cos(nose_phi))**2 + (a2 * np.sin(nose_phi))**2)
    dist_nose_center = np.hypot(nose_dx, nose_dy)
    n2e = np.maximum(0, dist_nose_center - r_nose_dir)
    
    # Same for tail
    tail_dx = tail_x - x2
    tail_dy = tail_y - y2
    tail_phi = np.arctan2(tail_dy, tail_dx) - th2
    r_tail_dir = (a2 * b2) / np.sqrt((b2 * np.cos(tail_phi))**2 + (a2 * np.sin(tail_phi))**2)
    dist_tail_center = np.hypot(tail_dx, tail_dy)
    t2e = np.maximum(0, dist_tail_center - r_tail_dir)
    
    return {
        'c2c': c2c,
        'facing_angle': facing_angle,
        'n2e': n2e,
        't2e': t2e,
        'vel': np.hypot(np.gradient(x1), np.gradient(y1))
    }

# Re-export feature builder that includes these new features
def build_heuristic_features(tracks: Dict[str, List[np.ndarray]], fly_idx: int) -> Dict[str, np.ndarray]:
    """
    Build features for heuristic classification.
    Assumes 2 flies. Partner is 1-fly_idx.
    """
    partner_idx = 1 - fly_idx
    
    # Ensure head/tail resolved
    # (Caller should have called resolve_head_tail once on tracks)
    
    feats = compute_social_features(tracks, fly_idx, partner_idx)
    v_par, v_perp = compute_component_velocities(tracks, fly_idx)
    feats['v_par'] = v_par
    feats['v_perp'] = v_perp
    feats['wing_angle'] = tracks['wing_angle'][fly_idx]
    feats['area'] = tracks['area'][fly_idx]
    
    return feats
