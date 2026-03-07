import numpy as np
from typing import List, Dict, Tuple
from .jab_parser import WeakLearner

def eval_weak_learner(learner: WeakLearner, X: np.ndarray) -> np.ndarray:
    """
    Evaluate a single weak learner on feature matrix X.
    X shape: (n_features, n_frames)
    learner.dim is 1-based index from MATLAB.
    """
    # MATLAB 1-based index -> Python 0-based
    idx = learner.dim - 1
    if idx < 0 or idx >= X.shape[0]:
        return np.zeros(X.shape[1], dtype=np.float32)
    
    feat_vals = X[idx, :]
    
    # boosting rule:
    # h(x) = alpha if direction*feat < direction*threshold else -alpha?
    # Usually:
    # if (x[dim] * direction) < (thr * direction): return alpha
    # else: return -alpha (or 0?)
    # GentleBoost / AdaBoost usually sum alpha.
    
    # JAABA standard:
    # h = (feat < thr) ? alpha : -alpha   (if dir=1 ? wait)
    # Let's verify standard.
    # Usually direction indicates which side is positive class.
    # If direction=1: feat < thr => positive? Or feat > thr?
    # Typically: h = alpha * sign(direction * (feat - thr)) ?
    
    # Re-reading JAABA C++ code or similar:
    # if (val < tr) match = true; else match = false;
    # if direction == 1: if match return alpha; else return -alpha; (So < means positive class)
    # if direction == -1: if match return -alpha; else return alpha; (So < means negative class -> > means positive)
    
    # Let's simplify:
    # pred = (val < tr)
    # score = pred * (direction * alpha) + (~pred) * (-direction * alpha)
    #       = direction * alpha * (2*pred - 1)
    
    pred = (feat_vals < learner.thr)
    score = (2.0 * pred - 1.0) * float(learner.direction) * learner.alpha
    return score.astype(np.float32)


def eval_boosting(learners: List[WeakLearner], X: np.ndarray) -> np.ndarray:
    """
    Evaluate strong classifier (sum of weak learners).
    """
    n_frames = X.shape[1]
    total_score = np.zeros(n_frames, dtype=np.float32)
    
    for learner in learners:
        total_score += eval_weak_learner(learner, X)
        
    return total_score


def scores_to_bouts(scores: np.ndarray, threshold: float = 0.0, min_len: int = 1, merge_gap: int = 0) -> List[Tuple[int, int]]:
    """
    Convert continuous scores to start/end indices of bouts.
    merge_gap: merge adjacent bouts separated by fewer than this many frames.
    """
    # Binary classification
    pred = scores > threshold

    # Find runs
    pred = np.concatenate(([False], pred, [False])) # Pad
    # diff != 0 gives edges
    diff = np.diff(pred.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1 # inclusive end

    bouts = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_len:
            bouts.append((s, e))

    # Merge bouts separated by small gaps
    if merge_gap > 0 and len(bouts) > 1:
        merged = [bouts[0]]
        for s, e in bouts[1:]:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= merge_gap:
                merged[-1] = (prev_s, e)
            else:
                merged.append((s, e))
        # Re-filter by min_len after merging
        bouts = [(s, e) for s, e in merged if (e - s + 1) >= min_len]

    return bouts

