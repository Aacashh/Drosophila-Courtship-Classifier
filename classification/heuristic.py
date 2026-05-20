from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from .inference import scores_to_bouts


def smooth_score(score: np.ndarray, fps: float, sigma_s: float = 0.2) -> np.ndarray:
    """Smooth a binary score array with a Gaussian filter, then re-threshold.
    Uses a wider window (0.2s) to bridge short tracking gaps and eliminate noise.
    Lower re-threshold (0.3) to preserve bouts interrupted by brief tracking dropout."""
    sigma_frames = max(1.0, sigma_s * fps)
    smoothed = gaussian_filter1d(score.astype(float), sigma=sigma_frames)
    return (smoothed > 0.3).astype(float)


def compute_courtship_index(bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
                            n_frames: int, fly_idx: int,
                            onset_frame: int = 0) -> float:
    """Compute Courtship Index: % of frames with ANY courtship behavior for a given fly.
    If onset_frame > 0, only count frames from onset_frame onward (exclude setup period)."""
    courtship_frames = np.zeros(n_frames, dtype=bool)
    for beh, bouts_list in bouts_by_behavior.items():
        if beh == 'Courtship':
            continue  # Don't double-count the unified Courtship entry
        if fly_idx < len(bouts_list):
            for start, end in bouts_list[fly_idx]:
                courtship_frames[start:end + 1] = True
    valid_frames = n_frames - onset_frame
    valid = valid_frames if valid_frames > 0 else 1
    return float(np.sum(courtship_frames[onset_frame:])) / valid * 100.0


DEFAULT_PARAMS = {
    'wing_ext_angle_deg': 55,
    'wing_ext_min_dur_s': 0.5,
    'cop_distance_mm': 3.0,
    'cop_velocity_mm_s': 0.5,
    'cop_min_dur_s': 5.0,
    'cop_stability_mm': 1.0,
    'follow_velocity_mm_s': 1.0,
    'follow_facing_deg': 60,
    'follow_dist_max_mm': 10.0,
    'follow_dist_min_mm': 1.5,
    'follow_min_dur_s': 0.5,
    'circling_lat_vel_mm_s': 1.5,
    'circling_dist_mm': 10.0,
    'circling_min_dur_s': 0.5,
    'att_cop_nose_dist_mm': 1.0,
    'att_cop_velocity_mm_s': 0.5,
    'att_cop_min_dur_s': 0.3,
    'merge_gap_s': 1.5,
    'courtship_merge_gap_s': 3.0,
    'courtship_min_dur_s': 0.5,
}


def _extract_overlap_mask(tracks: Dict) -> np.ndarray | None:
    if 'overlap' not in tracks:
        return None
    val = tracks['overlap']
    if isinstance(val, list):
        if len(val) == 0:
            return None
        return val[0]
    return val


def _compute_features(tracks: Dict[str, List[np.ndarray]], fps: float,
                      n_flies: int = 2) -> Dict:
    """Heavy step: resolve head/tail orientation and build per-fly features.

    Pure with respect to heuristic thresholds — the result can be cached and
    reused across many calls to `_apply_rules` with different parameter sets.
    """
    from tracking.features import build_heuristic_features, resolve_head_tail

    n_frames = len(tracks['x'][0])
    tracks = resolve_head_tail(tracks, n_flies, fps)
    feats = [build_heuristic_features(tracks, i, fps) for i in range(n_flies)]
    overlap_mask = _extract_overlap_mask(tracks)
    return {
        'feats': feats,
        'n_frames': n_frames,
        'n_flies': n_flies,
        'overlap_mask': overlap_mask,
    }


def _apply_rules(features: Dict, fps: float, px_per_mm: float,
                 params: dict = None) -> Dict[str, List[List[Tuple[int, int]]]]:
    """Cheap step: run rule thresholds against pre-computed features.

    Suitable for interactive slider tuning — called many times per second."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    feats = features['feats']
    n_flies = features['n_flies']
    n_frames = features['n_frames']
    overlap_mask = features['overlap_mask']
    merge = int(p['merge_gap_s'] * fps)

    results = {
        'WingExt': [[], []],
        'Copulation': [[], []],
        'Following': [[], []],
        'Circling': [[], []],
        'Attempted_Copulation': [[], []]
    }

    # --- Rules ---

    # 1. Wing Extension
    for i in range(n_flies):
        wing_angle = feats[i]['wing_angle']
        score = (wing_angle > np.radians(p['wing_ext_angle_deg'])).astype(float)
        if overlap_mask is not None:
            score[overlap_mask] = 0.0
        score = smooth_score(score, fps)
        results['WingExt'][i] = scores_to_bouts(score, min_len=int(p['wing_ext_min_dur_s'] * fps), merge_gap=merge)

    # 2. Copulation
    for i in range(n_flies):
        c2c_mm = feats[i]['c2c'] / px_per_mm
        vel_mm = feats[i]['vel'] / px_per_mm * fps

        is_close = c2c_mm < p['cop_distance_mm']
        is_slow = vel_mm < p['cop_velocity_mm_s']

        # Rolling std dev of c2c over 2-second window
        win = int(2.0 * fps)
        if win < 3:
            win = 3
        c2c_std = pd.Series(c2c_mm).rolling(win, center=True, min_periods=1).std().fillna(0).values
        is_stable = c2c_std < p['cop_stability_mm']

        score = (is_close & is_slow & is_stable).astype(float)
        if overlap_mask is not None:
            score[overlap_mask] = 0.0
        score = smooth_score(score, fps)
        results['Copulation'][i] = scores_to_bouts(score, min_len=int(p['cop_min_dur_s'] * fps), merge_gap=merge)

    # 3. Following
    for i in range(n_flies):
        c2c_mm = feats[i]['c2c'] / px_per_mm
        v_par_mm = feats[i]['v_par'] / px_per_mm * fps
        facing = np.abs(feats[i]['facing_angle'])

        score = (
            (v_par_mm > p['follow_velocity_mm_s']) &
            (facing < np.radians(p['follow_facing_deg'])) &
            (c2c_mm < p['follow_dist_max_mm']) &
            (c2c_mm > p['follow_dist_min_mm'])
        ).astype(float)
        if overlap_mask is not None:
            score[overlap_mask] = 0.0
        score = smooth_score(score, fps)
        results['Following'][i] = scores_to_bouts(score, min_len=int(p['follow_min_dur_s'] * fps), merge_gap=merge)

    # 4. Circling
    for i in range(n_flies):
        c2c_mm = feats[i]['c2c'] / px_per_mm
        v_perp_mm = feats[i]['v_perp'] / px_per_mm * fps
        facing = np.abs(feats[i]['facing_angle'])

        score = (
            (np.abs(v_perp_mm) > p['circling_lat_vel_mm_s']) &
            (c2c_mm < p['circling_dist_mm']) &
            (facing > np.radians(45)) &
            (facing < np.radians(135))
        ).astype(float)
        if overlap_mask is not None:
            score[overlap_mask] = 0.0
        score = smooth_score(score, fps)
        results['Circling'][i] = scores_to_bouts(score, min_len=int(p['circling_min_dur_s'] * fps), merge_gap=merge)

    # 5. Attempted Copulation
    for i in range(n_flies):
        n2e_mm = feats[i]['n2e'] / px_per_mm
        vel_mm = feats[i]['vel'] / px_per_mm * fps

        score = (
            (n2e_mm < p['att_cop_nose_dist_mm']) &
            (vel_mm > p['att_cop_velocity_mm_s'])
        ).astype(float)
        if overlap_mask is not None:
            score[overlap_mask] = 0.0
        score = smooth_score(score, fps)
        results['Attempted_Copulation'][i] = scores_to_bouts(score, min_len=int(p['att_cop_min_dur_s'] * fps), merge_gap=merge)

    # --- Mutual Exclusion (priority resolution) ---
    # Priority: Copulation > Attempted_Copulation > Following > Circling
    # WingExt is exempt (can co-occur with other behaviors)
    priority_behaviors = ['Copulation', 'Attempted_Copulation', 'Following', 'Circling']

    for i in range(n_flies):
        claimed = np.zeros(n_frames, dtype=bool)

        for beh in priority_behaviors:
            filtered_bouts = []
            for s, e in results[beh][i]:
                bout_frames = np.arange(s, e + 1)
                unclaimed = ~claimed[bout_frames]
                if np.any(unclaimed):
                    filtered_bouts.append((s, e))
                    claimed[s:e + 1] = True
            results[beh][i] = filtered_bouts

    # --- Unified Courtship (union of all sub-behaviors per fly) ---
    courtship_merge = int(p['courtship_merge_gap_s'] * fps)
    courtship_min = int(p['courtship_min_dur_s'] * fps)
    results['Courtship'] = [[], []]
    for i in range(n_flies):
        courtship_frames = np.zeros(n_frames, dtype=bool)
        for beh in results:
            if beh == 'Courtship':
                continue
            if i < len(results[beh]):
                for s, e in results[beh][i]:
                    courtship_frames[s:e + 1] = True
        courtship_score = courtship_frames.astype(float)
        results['Courtship'][i] = scores_to_bouts(
            courtship_score, threshold=0.0,
            min_len=courtship_min, merge_gap=courtship_merge)

    return results


class HeuristicClassifier:
    """Rule-based classifiers for courtship behaviors."""

    @staticmethod
    def classify(tracks: Dict[str, List[np.ndarray]], fps: float, px_per_mm: float,
                 params: dict = None) -> Dict[str, List[List[Tuple[int, int]]]]:
        """Returns { 'Behavior': [ [bouts_fly0], [bouts_fly1] ] }
        params: optional dict of threshold overrides (see DEFAULT_PARAMS)."""
        features = _compute_features(tracks, fps)
        return _apply_rules(features, fps, px_per_mm, params)
