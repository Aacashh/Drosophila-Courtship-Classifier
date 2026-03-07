from typing import Dict, List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .inference import scores_to_bouts


def smooth_score(score: np.ndarray, fps: float, sigma_s: float = 0.05) -> np.ndarray:
    """Smooth a binary score array with a Gaussian filter, then re-threshold at 0.5.
    Eliminates single-frame false positives."""
    sigma_frames = max(1.0, sigma_s * fps)
    smoothed = gaussian_filter1d(score.astype(float), sigma=sigma_frames)
    return (smoothed > 0.5).astype(float)


def compute_courtship_index(bouts_by_behavior: Dict[str, List[List[Tuple[int, int]]]],
                            n_frames: int, fly_idx: int) -> float:
    """Compute Courtship Index: % of frames with ANY courtship behavior for a given fly."""
    courtship_frames = np.zeros(n_frames, dtype=bool)
    for beh, bouts_list in bouts_by_behavior.items():
        if fly_idx < len(bouts_list):
            for start, end in bouts_list[fly_idx]:
                courtship_frames[start:end + 1] = True
    valid = n_frames if n_frames > 0 else 1
    return float(np.sum(courtship_frames)) / valid * 100.0


DEFAULT_PARAMS = {
    'wing_ext_angle_deg': 60,
    'wing_ext_min_dur_s': 1.0,
    'cop_distance_mm': 2.0,
    'cop_velocity_mm_s': 0.3,
    'cop_min_dur_s': 8.0,
    'cop_stability_mm': 0.5,
    'follow_velocity_mm_s': 2.0,
    'follow_facing_deg': 45,
    'follow_dist_max_mm': 8.0,
    'follow_dist_min_mm': 2.0,
    'follow_min_dur_s': 1.0,
    'circling_lat_vel_mm_s': 2.0,
    'circling_dist_mm': 10.0,
    'circling_min_dur_s': 1.0,
    'att_cop_nose_dist_mm': 0.5,
    'att_cop_velocity_mm_s': 1.0,
    'att_cop_min_dur_s': 0.5,
    'merge_gap_s': 0.5,
}


class HeuristicClassifier:
    """Rule-based classifiers for courtship behaviors."""

    @staticmethod
    def classify(tracks: Dict[str, List[np.ndarray]], fps: float, px_per_mm: float,
                 params: dict = None) -> Dict[str, List[List[Tuple[int, int]]]]:
        """Returns { 'Behavior': [ [bouts_fly0], [bouts_fly1] ] }
        params: optional dict of threshold overrides (see DEFAULT_PARAMS)."""
        p = {**DEFAULT_PARAMS, **(params or {})}
        n_flies = 2
        n_frames = len(tracks['x'][0])
        merge = int(p['merge_gap_s'] * fps)

        feats = [None, None]
        from tracking.features import build_heuristic_features, resolve_head_tail

        # Resolve orientation first
        tracks = resolve_head_tail(tracks, n_flies, fps)

        for i in range(n_flies):
            feats[i] = build_heuristic_features(tracks, i, fps)

        results = {
            'WingExt': [[], []],
            'Copulation': [[], []],
            'Following': [[], []],
            'Circling': [[], []],
            'Attempted_Copulation': [[], []]
        }

        # Check for overlap flag from tracker (zero out scores during overlaps)
        has_overlap = 'overlap' in tracks and len(tracks['overlap']) > 0
        overlap_mask = None
        if has_overlap:
            overlap_mask = tracks['overlap'][0] if isinstance(tracks['overlap'], list) else tracks['overlap']

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
            c2c_std = np.array([
                np.std(c2c_mm[max(0, t - win // 2):t + win // 2 + 1])
                for t in range(n_frames)
            ])
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
            # Build a frame-level mask of which behavior "owns" each frame
            claimed = np.zeros(n_frames, dtype=bool)

            for beh in priority_behaviors:
                filtered_bouts = []
                for s, e in results[beh][i]:
                    # Check if any frame in this bout is already claimed by higher priority
                    bout_frames = np.arange(s, e + 1)
                    unclaimed = ~claimed[bout_frames]
                    if np.any(unclaimed):
                        # Keep the bout but mark its frames as claimed
                        filtered_bouts.append((s, e))
                        claimed[s:e + 1] = True
                results[beh][i] = filtered_bouts

        return results
