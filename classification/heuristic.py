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
        # RESEARCH FIX: Threshold increased to 50 deg (approx 0.87 rad)
        for i in range(n_flies):
            wing_angle = feats[i]['wing_angle']
            score = (wing_angle > np.radians(50)).astype(float)
            results['WingExt'][i] = scores_to_bouts(score, min_len=int(0.1 * fps))

        # 2. Copulation
        # RESEARCH FIX: C2C < 2.5mm, Vel < 0.5 mm/s (strict stationarity)
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            vel_mm = feats[i]['vel'] / px_per_mm * fps
            
            is_close = c2c_mm < 2.5 
            is_slow = vel_mm < 0.5
            
            # Copulation is a mutual state, usually
            score = (is_close & is_slow).astype(float)
            results['Copulation'][i] = scores_to_bouts(score, min_len=int(5.0 * fps))
            
        # 3. Following
        # RESEARCH FIX: Facing < 45 deg, C2C < 8mm (tightened from 10)
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            vel_mm = feats[i]['vel'] / px_per_mm * fps # Note: using vel_mm instead of v_par_mm as per logic, or sticking to v_par logic if that was better? 
            # Original code used v_par_mm. User request says "vel > 2.0" in text but "vel_mm" in snippet. 
            # However, snippet uses `vel_mm > 2.0` in the "Following" section description but in code uses `v_par_mm`. 
            # Wait, the user snippet for Following says:
            # score = ((vel_mm > 2.0) & ...
            # But the original code used v_par_mm. 
            # I will follow the user provided snippet logic which uses vel_mm or check if I should preserve v_par.
            # The snippet provided by user:
            # score = ((vel_mm > 2.0) & (facing < np.radians(45)) & (c2c_mm < 8.0) & (c2c_mm > 2.0)).
            # I will stick to what the user provided in the snippet.
            
            facing = np.abs(feats[i]['facing_angle'])
            
            score = (
                (vel_mm > 2.0) & 
                (facing < np.radians(45)) & 
                (c2c_mm < 8.0) &
                (c2c_mm > 2.0) # Not touching
            ).astype(float)
            
            results['Following'][i] = scores_to_bouts(score, min_len=int(0.5 * fps))
            
        # 4. Circling
        # RESEARCH FIX: Must use Lateral Velocity > 2mm/s
        for i in range(n_flies):
            c2c_mm = feats[i]['c2c'] / px_per_mm
            # We need to make sure we use the lateral velocity.
            # The snippet uses: lat_vel_mm = np.abs(lat_vel) / px_per_mm * fps
            # But we have feats[i]['v_perp'].
            v_perp_mm = feats[i]['v_perp'] / px_per_mm * fps
            facing = np.abs(feats[i]['facing_angle'])
            
            score = (
                (np.abs(v_perp_mm) > 2.0) &
                (c2c_mm < 10.0) &
                (facing > np.radians(45)) &
                (facing < np.radians(135)) # Side facing
            ).astype(float)
             
            results['Circling'][i] = scores_to_bouts(score, min_len=int(0.3 * fps))
            
        # 5. Attempted Copulation
        # RESEARCH FIX: N2E < 0.5mm (Touching)
        for i in range(n_flies):
            n2e_mm = feats[i]['n2e'] / px_per_mm
            vel_mm = feats[i]['vel'] / px_per_mm * fps
            
            score = (
                (n2e_mm < 0.5) &
                (vel_mm > 1.0) # Active
            ).astype(float)
            
            results['Attempted_Copulation'][i] = scores_to_bouts(score, min_len=int(0.2 * fps))

        return results

