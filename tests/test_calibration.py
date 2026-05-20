"""Tests for evaluation.calibration."""

import numpy as np

from evaluation.calibration import estimate_px_per_mm


def _make_tracks(a_values_per_fly):
    return {"a": [np.array(v, dtype=float) for v in a_values_per_fly]}


class TestEstimatePxPerMm:
    def test_basic_two_flies(self):
        # Both flies have semi-major = 15 px. Body length = 2.5mm -> 12 px/mm.
        tracks = _make_tracks([[15.0] * 100, [15.0] * 100])
        result = estimate_px_per_mm(tracks)
        assert result == 12.0

    def test_custom_body_length(self):
        tracks = _make_tracks([[20.0] * 50, [20.0] * 50])
        # body length 4mm -> semi-major 2mm -> 10 px/mm
        result = estimate_px_per_mm(tracks, assumed_body_length_mm=4.0)
        assert result == 10.0

    def test_ignores_nans_and_nonpositive(self):
        a = np.array([np.nan, 0, -1, 15, 15, 15], dtype=float)
        tracks = {"a": [a]}
        result = estimate_px_per_mm(tracks)
        assert result == 12.0  # median of {15, 15, 15} = 15

    def test_returns_none_when_no_data(self):
        assert estimate_px_per_mm({"a": []}) is None
        assert estimate_px_per_mm({}) is None
        empty = {"a": [np.array([np.nan, np.nan])]}
        assert estimate_px_per_mm(empty) is None

    def test_returns_none_for_invalid_body_length(self):
        tracks = _make_tracks([[15.0] * 10])
        assert estimate_px_per_mm(tracks, assumed_body_length_mm=0) is None
