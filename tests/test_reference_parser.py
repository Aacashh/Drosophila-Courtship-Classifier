"""Tests for evaluation.reference_parser."""

from pathlib import Path

import pytest

from evaluation.reference_parser import (
    ReferenceChamber,
    parse_reference_csv,
    parse_time_string,
)


class TestParseTimeString:
    def test_h_mm_ss(self):
        assert parse_time_string("0:00:07") == 7.0
        assert parse_time_string("1:23:45") == 5025.0
        assert parse_time_string("0:10:00") == 600.0

    def test_m_ss(self):
        assert parse_time_string("1:23") == 83.0
        assert parse_time_string("10:00") == 600.0

    def test_blank_and_nan(self):
        assert parse_time_string("") is None
        assert parse_time_string("   ") is None
        assert parse_time_string("nan") is None
        assert parse_time_string("NaN") is None
        assert parse_time_string(None) is None

    def test_garbage(self):
        assert parse_time_string("0:07:35:00") is None
        assert parse_time_string("foo") is None
        assert parse_time_string("3") is None


def _write_csv(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "ref.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestParseReferenceCsv:
    def test_basic_two_chambers(self, tmp_path):
        csv = _write_csv(tmp_path, [
            ",,,",
            ",,,",
            "Chamber,Start,End,Duration",
            ",,,",
            "01 (bottom right),0:00:07,0:00:24,0:00:17",
            ",0:00:26,0:00:30,0:00:04",
            ",,,0:00:21",
            ",,,0",
            ",,,",
            '"02 (top middle)",0:01:00,0:02:00,0:01:00',
            ",0:03:00,0:04:00,0:01:00",
            ",,,0:02:00",
        ])
        chambers = parse_reference_csv(csv)
        assert len(chambers) == 2
        assert chambers[0].descriptor == "01 (bottom right)"
        assert chambers[0].bouts == [(7.0, 24.0), (26.0, 30.0)]
        assert chambers[1].descriptor == "02 (top middle)"
        assert chambers[1].bouts == [(60.0, 120.0), (180.0, 240.0)]

    def test_skips_padding_rows(self, tmp_path):
        csv = _write_csv(tmp_path, [
            ",,,",
            ",,,",
            ",,,",
            ",,,",
            "Chamber,Start,End,Duration",
            "01 (test),0:00:05,0:00:10,0:00:05",
        ])
        chambers = parse_reference_csv(csv)
        assert len(chambers) == 1
        assert chambers[0].bouts == [(5.0, 10.0)]

    def test_filters_footer_summary_rows(self, tmp_path):
        csv = _write_csv(tmp_path, [
            "Chamber,Start,End,Duration",
            "01 (bottom right),0:00:07,0:00:24,0:00:17",
            ",,,0:07:35",
            ",,,0",
        ])
        chambers = parse_reference_csv(csv)
        assert chambers[0].bouts == [(7.0, 24.0)]

    def test_rejects_inverted_bouts(self, tmp_path):
        csv = _write_csv(tmp_path, [
            "Chamber,Start,End,Duration",
            "01 (test),0:00:20,0:00:10,0:00:00",
            ",0:00:30,0:00:40,0:00:10",
        ])
        chambers = parse_reference_csv(csv)
        assert chambers[0].bouts == [(30.0, 40.0)]

    def test_real_reference_file(self):
        ref = Path(__file__).resolve().parents[1] / "reference" / "Sexy times (18+)  - 4D CTRL GBX.csv"
        if not ref.exists():
            pytest.skip("reference CSV not present")
        chambers = parse_reference_csv(ref)
        assert len(chambers) == 7
        descriptors = [c.descriptor for c in chambers]
        assert descriptors[0].startswith("01")
        assert "bottom right" in descriptors[0].lower()
        assert descriptors[-1].startswith("07")
        for chamber in chambers:
            for start, end in chamber.bouts:
                assert end > start
                assert start >= 0
                assert end <= 600 + 1  # 10-min video, allow tiny overrun

    def test_total_seconds(self):
        chamber = ReferenceChamber(descriptor="x", bouts=[(0.0, 10.0), (20.0, 25.0)])
        assert chamber.total_seconds == 15.0
