"""Tests for evaluation.spatial_mapper."""

import json

import pytest

from evaluation.reference_parser import ReferenceChamber
from evaluation.spatial_mapper import (
    ParsedDescriptor,
    assign_grid_coordinates,
    build_mapping,
    match_descriptor_to_chamber,
    parse_descriptor,
    save_mapping_override,
)


class TestParseDescriptor:
    @pytest.mark.parametrize("raw, row_key, row_offset, col_key", [
        ("01 (bottom right)", "bottom", 0, "right"),
        ("02 (second row from the bottom, right)", "bottom", 1, "right"),
        ("03 (second row from bottom, middle)", "bottom", 1, "middle"),
        ("04 (second row from the top, right)", "top", 1, "right"),
        ("05 (second row from top, middle)", "top", 1, "middle"),
        ("06 (top row, right)", "top", 0, "right"),
        ("07 (top row middle)", "top", 0, "middle"),
    ])
    def test_real_reference_descriptors(self, raw, row_key, row_offset, col_key):
        parsed = parse_descriptor(raw)
        assert parsed.row_key == row_key
        assert parsed.row_offset == row_offset
        assert parsed.col_key == col_key

    def test_center_aliases_middle(self):
        parsed = parse_descriptor("(top row, center)")
        assert parsed.col_key == "middle"
        parsed = parse_descriptor("(bottom centre)")
        assert parsed.col_key == "middle"

    def test_bare_keywords(self):
        parsed = parse_descriptor("bottom left")
        assert parsed.row_key == "bottom"
        assert parsed.col_key == "left"

    def test_no_row_no_col(self):
        parsed = parse_descriptor("foo bar")
        assert parsed.row_key is None
        assert parsed.col_key is None
        assert not parsed.is_resolved()


class TestAssignGridCoordinates:
    def test_simple_2x3(self):
        # bboxes: (x, y, w, h) — two rows of three
        bboxes = [
            (10, 10, 50, 50),    # top-left
            (70, 10, 50, 50),    # top-middle
            (130, 10, 50, 50),   # top-right
            (10, 80, 50, 50),    # bottom-left
            (70, 80, 50, 50),    # bottom-middle
            (130, 80, 50, 50),   # bottom-right
        ]
        coords = assign_grid_coordinates(bboxes)
        assert coords == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def test_unsorted_input(self):
        # 2x2 grid with bboxes deliberately out of reading order
        bboxes = [
            (130, 80, 50, 50),   # idx 0 -> bottom-right (row 1, col 1)
            (10, 10, 50, 50),    # idx 1 -> top-left     (row 0, col 0)
            (70, 80, 50, 50),    # idx 2 -> bottom-left  (row 1, col 0)
            (70, 10, 50, 50),    # idx 3 -> top-right    (row 0, col 1)
        ]
        coords = assign_grid_coordinates(bboxes)
        assert coords[0] == (1, 1)
        assert coords[1] == (0, 0)
        assert coords[2] == (1, 0)
        assert coords[3] == (0, 1)

    def test_missing_chamber_in_grid(self):
        # 2x3 grid with the middle-bottom chamber missing
        bboxes = [
            (10, 10, 50, 50),
            (70, 10, 50, 50),
            (130, 10, 50, 50),
            (10, 80, 50, 50),
            (130, 80, 50, 50),
        ]
        coords = assign_grid_coordinates(bboxes)
        # Bottom row should have 2 chambers, column indices 0 and 1 (left-most pair)
        assert coords[3] == (1, 0)
        assert coords[4] == (1, 1)

    def test_empty(self):
        assert assign_grid_coordinates([]) == []


class TestMatchDescriptor:
    def setup_method(self):
        # 4x3 grid
        self.bboxes = []
        for r in range(4):
            for c in range(3):
                self.bboxes.append((c * 60 + 10, r * 60 + 10, 50, 50))
        self.coords = assign_grid_coordinates(self.bboxes)

    def test_bottom_right(self):
        desc = ParsedDescriptor("bottom", 0, "right", "")
        idx = match_descriptor_to_chamber(desc, self.coords)
        # bottom row = 3, right column = col 2 -> bbox 3*3 + 2 = 11
        assert idx == 11

    def test_top_left(self):
        desc = ParsedDescriptor("top", 0, "left", "")
        assert match_descriptor_to_chamber(desc, self.coords) == 0

    def test_second_from_top_middle(self):
        desc = ParsedDescriptor("top", 1, "middle", "")
        # row 1, col 1 -> idx 1*3 + 1 = 4
        assert match_descriptor_to_chamber(desc, self.coords) == 4

    def test_second_from_bottom_right(self):
        desc = ParsedDescriptor("bottom", 1, "right", "")
        # row 2, col 2 -> idx 2*3 + 2 = 8
        assert match_descriptor_to_chamber(desc, self.coords) == 8

    def test_missing_column_in_singleton_row_ok(self):
        # 1x1 grid: a row of length 1 doesn't need a column hint
        single = assign_grid_coordinates([(0, 0, 10, 10)])
        desc = ParsedDescriptor("top", 0, None, "")
        assert match_descriptor_to_chamber(desc, single) == 0

    def test_missing_row_returns_none(self):
        desc = ParsedDescriptor(None, 0, "right", "")
        assert match_descriptor_to_chamber(desc, self.coords) is None

    def test_out_of_range_offset(self):
        # offset=10 from top in a 4-row grid -> out of range
        desc = ParsedDescriptor("top", 10, "right", "")
        assert match_descriptor_to_chamber(desc, self.coords) is None


class TestBuildMapping:
    def test_full_reference_layout(self):
        # Simulate a 4-row grid that contains all 7 reference descriptors.
        # Row 0 (top) needs: 06 right, 07 middle
        # Row 1 (second from top): 04 right, 05 middle
        # Row 2 (second from bottom): 02 right, 03 middle
        # Row 3 (bottom): 01 right
        bboxes = []
        for r in range(4):
            for c in range(3):
                bboxes.append((c * 60 + 10, r * 60 + 10, 50, 50))

        ref = [
            ReferenceChamber(descriptor="01 (bottom right)"),
            ReferenceChamber(descriptor="02 (second row from the bottom, right)"),
            ReferenceChamber(descriptor="03 (second row from bottom, middle)"),
            ReferenceChamber(descriptor="04 (second row from the top, right)"),
            ReferenceChamber(descriptor="05 (second row from top, middle)"),
            ReferenceChamber(descriptor="06 (top row, right)"),
            ReferenceChamber(descriptor="07 (top row middle)"),
        ]
        mapping, ambig = build_mapping(bboxes, ref)
        assert ambig == []
        assert mapping["01 (bottom right)"] == 11  # row 3 col 2
        assert mapping["07 (top row middle)"] == 1  # row 0 col 1

    def test_override_wins(self, tmp_path):
        bboxes = [(0, 0, 10, 10), (20, 0, 10, 10)]
        ref = [ReferenceChamber(descriptor="01 (top right)")]
        override = tmp_path / "override.json"
        override.write_text(json.dumps({"01 (top right)": 0}))
        mapping, _ = build_mapping(bboxes, ref, override_path=override)
        assert mapping["01 (top right)"] == 0  # would auto-resolve to 1

    def test_ambiguity_reported(self):
        # Descriptor that can't be parsed
        bboxes = [(0, 0, 10, 10)]
        ref = [ReferenceChamber(descriptor="mystery chamber")]
        mapping, ambig = build_mapping(bboxes, ref)
        assert mapping["mystery chamber"] is None
        assert "mystery chamber" in ambig

    def test_save_override_roundtrip(self, tmp_path):
        path = tmp_path / "saved.json"
        mapping = {"a": 0, "b": None, "c": 5}
        save_mapping_override(mapping, path)
        data = json.loads(path.read_text())
        assert data == {"a": 0, "c": 5}
