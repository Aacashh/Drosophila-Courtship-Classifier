"""Tests for evaluation.compare."""

import csv
from pathlib import Path

import numpy as np
import pytest

from evaluation.compare import (
    BoutMetrics,
    FrameMetrics,
    bout_level_metrics,
    bouts_to_frame_labels,
    evaluate_chamber,
    frame_level_metrics,
    load_predicted_courtship,
)


class TestBoutsToFrameLabels:
    def test_basic(self):
        labels = bouts_to_frame_labels([(0.0, 1.0), (2.0, 3.0)], n_frames=120, fps=30.0)
        assert labels.shape == (120,)
        assert labels[:30].all()
        assert not labels[30:60].any()
        assert labels[60:90].all()
        assert not labels[90:].any()

    def test_empty(self):
        labels = bouts_to_frame_labels([], 100, 30.0)
        assert not labels.any()

    def test_clipped_to_n_frames(self):
        labels = bouts_to_frame_labels([(0.0, 100.0)], n_frames=10, fps=1.0)
        assert labels.shape == (10,)
        assert labels.all()


class TestFrameLevelMetrics:
    def test_perfect_overlap(self):
        bouts = [(0.0, 1.0), (2.0, 3.0)]
        m = frame_level_metrics(bouts, bouts, n_frames=120, fps=30.0)
        assert m.f1 == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.iou == 1.0

    def test_disjoint(self):
        m = frame_level_metrics([(0.0, 1.0)], [(2.0, 3.0)], n_frames=120, fps=30.0)
        assert m.f1 == 0.0
        assert m.iou == 0.0

    def test_partial_overlap(self):
        # Pred [0,1], Ref [0.5, 1.5]. Overlap = 0.5s, union = 1.5s -> IoU = 1/3.
        m = frame_level_metrics([(0.0, 1.0)], [(0.5, 1.5)], n_frames=60, fps=30.0)
        # tp=15, fp=15, fn=15
        assert m.tp == 15 and m.fp == 15 and m.fn == 15
        assert m.precision == pytest.approx(0.5)
        assert m.recall == pytest.approx(0.5)
        assert m.iou == pytest.approx(15 / 45)

    def test_empty_pred_and_ref(self):
        m = frame_level_metrics([], [], n_frames=100, fps=30.0)
        assert m.f1 == 0.0
        assert m.tn == 100


class TestBoutLevelMetrics:
    def test_perfect_match(self):
        bouts = [(0.0, 1.0), (2.0, 3.0)]
        m = bout_level_metrics(bouts, bouts, iou_threshold=0.3)
        assert m.matched == 2
        assert m.false_positives == 0
        assert m.missed == 0
        assert m.median_iou == 1.0

    def test_misses_and_false_positives(self):
        pred = [(0.0, 1.0), (10.0, 11.0)]
        ref = [(0.0, 1.0), (20.0, 21.0)]
        m = bout_level_metrics(pred, ref, iou_threshold=0.3)
        assert m.matched == 1
        assert m.false_positives == 1
        assert m.missed == 1

    def test_greedy_picks_highest_iou(self):
        # Two ref bouts close together; one pred should match the higher-IoU one
        pred = [(0.0, 1.0)]
        ref = [(0.0, 0.9), (0.0, 1.0)]
        m = bout_level_metrics(pred, ref, iou_threshold=0.3)
        assert m.matched == 1
        assert m.median_iou == pytest.approx(1.0)

    def test_below_threshold_unmatched(self):
        pred = [(0.0, 1.0)]
        ref = [(0.0, 0.1)]  # IoU = 0.1, below 0.3
        m = bout_level_metrics(pred, ref, iou_threshold=0.3)
        assert m.matched == 0
        assert m.false_positives == 1
        assert m.missed == 1


class TestLoadPredictedCourtship:
    def _write_csv(self, path: Path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Behavior", "Fly", "Start_Time (s)", "End_Time (s)",
                "Duration (s)", "Start_Frame", "End_Frame",
            ])
            writer.writeheader()
            writer.writerows(rows)

    def test_male_only_filter(self, tmp_path):
        path = tmp_path / "chamber_0_results.csv"
        self._write_csv(path, [
            {"Behavior": "Courtship", "Fly": "Fly 0 (Male)",
             "Start_Time (s)": "1.0", "End_Time (s)": "2.0",
             "Duration (s)": "1.0", "Start_Frame": 30, "End_Frame": 60},
            {"Behavior": "Courtship", "Fly": "Fly 1 (Female)",
             "Start_Time (s)": "5.0", "End_Time (s)": "6.0",
             "Duration (s)": "1.0", "Start_Frame": 150, "End_Frame": 180},
            {"Behavior": "WingExt", "Fly": "Fly 0 (Male)",
             "Start_Time (s)": "0.5", "End_Time (s)": "0.8",
             "Duration (s)": "0.3", "Start_Frame": 15, "End_Frame": 24},
        ])
        bouts, scope = load_predicted_courtship(path, male_only=True)
        assert scope == "male"
        assert bouts == [(1.0, 2.0)]

    def test_falls_back_to_union_when_no_male(self, tmp_path):
        path = tmp_path / "chamber_0_results.csv"
        self._write_csv(path, [
            {"Behavior": "Courtship", "Fly": "Fly 0",
             "Start_Time (s)": "1.0", "End_Time (s)": "2.0",
             "Duration (s)": "1.0", "Start_Frame": 30, "End_Frame": 60},
            {"Behavior": "Courtship", "Fly": "Fly 1",
             "Start_Time (s)": "1.5", "End_Time (s)": "3.0",
             "Duration (s)": "1.5", "Start_Frame": 45, "End_Frame": 90},
        ])
        bouts, scope = load_predicted_courtship(path, male_only=True)
        assert scope == "union"
        # Overlapping bouts get merged
        assert bouts == [(1.0, 3.0)]

    def test_empty_courtship(self, tmp_path):
        path = tmp_path / "chamber_0_results.csv"
        self._write_csv(path, [
            {"Behavior": "WingExt", "Fly": "Fly 0 (Male)",
             "Start_Time (s)": "0.5", "End_Time (s)": "0.8",
             "Duration (s)": "0.3", "Start_Frame": 15, "End_Frame": 24},
        ])
        bouts, scope = load_predicted_courtship(path)
        assert bouts == []
        assert scope == "empty"


class TestEvaluateChamber:
    def test_aggregates_frame_and_bout(self):
        pred = [(0.0, 1.0)]
        ref = [(0.0, 1.0), (5.0, 6.0)]
        result = evaluate_chamber(pred, ref, n_frames=300, fps=30.0)
        assert result["n_pred_bouts"] == 1
        assert result["n_ref_bouts"] == 2
        assert result["bout_matched"] == 1
        assert result["bout_missed"] == 1
        assert result["frame_recall"] == pytest.approx(30 / 60)
