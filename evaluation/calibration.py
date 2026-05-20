"""Auto-calibrate `px_per_mm` from tracked fly body length.

The tracker fits an ellipse to each fly and stores the semi-major axis in
`tracks['a']` (centroid -> head distance, in pixels). For Drosophila this
maps to roughly half the body length, so we can back out the pixel scale
given a literature-derived body-length prior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


DEFAULT_BODY_LENGTH_MM = 2.5  # adult Drosophila melanogaster, typical


def estimate_px_per_mm(
    tracks: Dict[str, List[np.ndarray]],
    assumed_body_length_mm: float = DEFAULT_BODY_LENGTH_MM,
) -> Optional[float]:
    """Estimate px/mm from the median semi-major axis across both flies.

    Returns None if the tracks dictionary has no usable `a` data — calling
    code should fall back to a hardcoded value (e.g. the sidebar default).
    """
    if "a" not in tracks or not tracks["a"]:
        return None
    if assumed_body_length_mm <= 0:
        return None

    samples: List[np.ndarray] = []
    for arr in tracks["a"]:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a) & (a > 0)]
        if a.size > 0:
            samples.append(a)
    if not samples:
        return None

    median_a_px = float(np.median(np.concatenate(samples)))
    if median_a_px <= 0:
        return None

    # semi-major ~= body_length / 2
    return median_a_px / (assumed_body_length_mm / 2.0)
