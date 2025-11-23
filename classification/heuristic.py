from typing import Dict, List, Tuple
import numpy as np
from .inference import scores_to_bouts

class HeuristicClassifier:
    """
    Rule-based classifiers for courtship behaviors.
    """
    
    @staticmethod
    def classify(tracks: Dict[str, List[np.ndarray]], fps: float, px_per_mm: float) -> Dict[str, List[List[Tuple[int, int]]]]:
        """
        Returns { 'Behavior': [ [bouts_fly0], [bouts_fly1] ] }
        """
        n_flies = 2
        n_frames = len(tracks['x'][0])
        
        # Metrics for both flies
        # We need social features for both
        feats = [None, None]
        from tracking.features import build_heuristic_features, resolve_head_tail
        
        # Resolve orientation first
        tracks = resolve_head_tail(tracks, n_flies)
        
        for i in range(n_flies):
            feats[i] = build_heuristic_features(tracks, i)
            
        results = {
            'WingExt': [[], []],
            'Copulation': [[], []],
            'Following': [[], []],
            'Circling': [[], []],
            'Attempted_Copulation': [[], []]
        }
        
        # --- Rules ---
        
        # 1. Wing Extension
        # Rule: Wing angle > 30 degrees (pi/6)
        # Duration > 0.1s ?
        for i in range(n_flies):
            wing_angle = feats[i]['wing_angle']
            score = (wing_angle > (np.pi / 6)).astype(float)
            results['WingExt'][i] = scores_to_bouts(score, min_len=int(0.05 * fps))

        # 2. Copulation
        # Rule: Very close, low velocity, long duration.
        # c2c < 2.5 mm (approx body length)
        # vel < 1 mm/s
        # Duration > 10s? (Copulation is usually minutes, but let's detect starts)
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            vel_mm = feats[i]['vel'] / px_per_mm * fps
            
            is_close = c2c_mm < 3.0 
            is_slow = vel_mm < 2.0
            
            # Copulation is a mutual state, usually
            score = (is_close & is_slow).astype(float)
            results['Copulation'][i] = scores_to_bouts(score, min_len=int(5.0 * fps))
            
        # 3. Following
        # Rule: Moving Forward, Facing Partner, Close
        # v_par > 2 mm/s
        # abs(facing_angle) < 45 deg
        # c2c < 8 mm
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            v_par_mm = feats[i]['v_par'] / px_per_mm * fps
            facing = np.abs(feats[i]['facing_angle'])
            
            score = (
                (v_par_mm > 2.0) & 
                (facing < np.radians(45)) & 
                (c2c_mm < 10.0) &
                (c2c_mm > 2.0) # Not touching
            ).astype(float)
            
            results['Following'][i] = scores_to_bouts(score, min_len=int(0.5 * fps))
            
        # 4. Circling
        # Rule: Significant lateral movement
        # |v_perp| > 2.0 mm/s
        # Facing roughly partner (90 deg?)
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            v_perp_mm = feats[i]['v_perp'] / px_per_mm * fps
            facing = np.abs(feats[i]['facing_angle'])
            
            score = (
                (np.abs(v_perp_mm) > 2.0) &
                (c2c_mm < 10.0) &
                (facing > np.radians(45)) &
                (facing < np.radians(135)) # Side facing
            ).astype(float)
             
            results['Circling'][i] = scores_to_bouts(score, min_len=int(0.5 * fps))
            
        # 5. Attempted Copulation
        # Rule: Male touches female tail with nose, but no copulation?
        # n2e < 1 mm
        # t2e (partner) ?
        # Usually: Nose of male near Tail of female.
        # Female tail is ... well, ellipse is symmetric in our tracker.
        # But we have t2e and n2e.
        # If Male Nose is close to Female (any part?), and Female is moving away?
        # Let's just use: n2e < 0.5 mm AND not Copulating.
        for i in range(n_flies):
            n2e_mm = feats[i]['n2e'] / px_per_mm
            vel_mm = feats[i]['vel'] / px_per_mm * fps
            
            score = (
                (n2e_mm < 1.0) &
                (vel_mm > 1.0) # Active
            ).astype(float)
            
            results['Attempted_Copulation'][i] = scores_to_bouts(score, min_len=int(0.2 * fps))

        return results

