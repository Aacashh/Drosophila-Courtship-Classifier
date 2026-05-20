"""Map positional descriptors (e.g. `02 (second row from the bottom, right)`)
to detected chamber indices, fully data-driven so the same code works for any
arena grid the chamber detector returns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .reference_parser import ReferenceChamber


# Ordinal phrases like "second from bottom" map to an offset from the edge.
_ORDINAL_OFFSETS = {
    "first": 0, "1st": 0, "one": 0,
    "second": 1, "2nd": 1, "two": 1,
    "third": 2, "3rd": 2, "three": 2,
    "fourth": 3, "4th": 3, "four": 3,
    "fifth": 4, "5th": 4, "five": 4,
}

_ROW_KEYWORDS = {"top", "bottom", "middle"}
_COL_KEYWORDS = {"left", "middle", "right", "center", "centre"}

_ORDINAL_PATTERN = re.compile(
    r"(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+(?:row\s+)?"
    r"(?:from\s+(?:the\s+)?)?(top|bottom)"
)


@dataclass
class ParsedDescriptor:
    """Structured row + column anchor parsed from a free-text descriptor."""
    row_key: Optional[str]         # 'top' | 'bottom' | 'middle' | None
    row_offset: int                # how many rows away from the anchor edge
    col_key: Optional[str]         # 'left' | 'middle' | 'right' | None
    raw: str

    def is_resolved(self) -> bool:
        return self.row_key is not None or self.col_key is not None


def _strip_prefix_and_parens(text: str) -> str:
    """Drop leading `NN` identifier and unwrap parenthesized content."""
    cleaned = text.strip()
    # Remove a leading numeric tag if a parenthesis follows it
    cleaned = re.sub(r"^\d+\s*", "", cleaned)
    # If the body is wrapped in parentheses, pull it out
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    # If there's a "(...)" suffix anywhere, prefer its contents
    paren_match = re.search(r"\(([^()]+)\)", cleaned)
    if paren_match:
        cleaned = paren_match.group(1)
    return cleaned.strip().lower()


def parse_descriptor(text: str) -> ParsedDescriptor:
    """Parse a positional descriptor string into row/column anchors.

    Examples:
        `01 (bottom right)`                       -> row='bottom' off=0, col='right'
        `02 (second row from the bottom, right)`  -> row='bottom' off=1, col='right'
        `05 (second row from top, middle)`        -> row='top' off=1, col='middle'
        `07 (top row middle)`                     -> row='top' off=0, col='middle'
    """
    body = _strip_prefix_and_parens(text)

    row_key: Optional[str] = None
    row_offset = 0
    col_key: Optional[str] = None

    ordinal = _ORDINAL_PATTERN.search(body)
    if ordinal:
        ord_word, edge = ordinal.groups()
        row_offset = _ORDINAL_OFFSETS.get(ord_word, 0)
        row_key = edge
    else:
        # Prefer "<X> row" phrasing over a bare keyword, so "top row middle"
        # picks row='top' (not 'middle' as a row anchor).
        row_phrase = re.search(r"\b(top|bottom|middle)\s+row\b", body)
        if row_phrase:
            row_key = row_phrase.group(1)
            row_offset = 0
        else:
            # Plain keyword: top/bottom win over middle.
            for kw in ("top", "bottom", "middle"):
                if re.search(rf"\b{kw}\b", body):
                    row_key = kw
                    row_offset = 0
                    break

    # Column keyword — prefer the last occurrence so trailing context wins
    # (e.g. "second row from the bottom, right" -> column is "right").
    col_matches = [kw for kw in _COL_KEYWORDS if re.search(rf"\b{kw}\b", body)]
    if col_matches:
        # Find the keyword latest in the text
        last_col = None
        last_pos = -1
        for kw in col_matches:
            pos = body.rfind(kw)
            if pos > last_pos:
                last_pos = pos
                last_col = kw
        if last_col in {"center", "centre"}:
            col_key = "middle"
        else:
            col_key = last_col
        # If only "middle" was found and it's the same as the row keyword,
        # ignore it as a column hint.
        if col_key == "middle" and row_key == "middle":
            col_key = None

    return ParsedDescriptor(row_key=row_key, row_offset=row_offset,
                            col_key=col_key, raw=text)


def assign_grid_coordinates(
    bboxes: Sequence[Tuple[int, int, int, int]],
) -> List[Tuple[int, int]]:
    """Group bboxes into rows by y-centroid, return (row_idx, col_idx) per bbox.

    Rows are clustered with a tolerance of `median_h * 0.5` — same heuristic
    that `utils.video.detect_chambers` uses to sort in reading order. Within
    each row, bboxes are ordered left-to-right by x-centroid.
    """
    if not bboxes:
        return []

    centers = [(x + w / 2.0, y + h / 2.0, idx) for idx, (x, y, w, h) in enumerate(bboxes)]
    heights = [h for _, _, w, h in [(b[0], b[1], b[2], b[3]) for b in bboxes]]
    median_h = sorted(heights)[len(heights) // 2] if heights else 1.0
    row_tol = max(1.0, median_h * 0.5)

    # Sort by y, then cluster into rows by tolerance
    by_y = sorted(centers, key=lambda c: c[1])
    rows: List[List[Tuple[float, float, int]]] = []
    for cx, cy, idx in by_y:
        if rows and abs(cy - rows[-1][0][1]) <= row_tol:
            rows[-1].append((cx, cy, idx))
        else:
            rows.append([(cx, cy, idx)])

    # Sort each row left-to-right and emit (row_idx, col_idx) per original bbox idx
    coords: Dict[int, Tuple[int, int]] = {}
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda c: c[0])
        for col_idx, (_, _, original_idx) in enumerate(row_sorted):
            coords[original_idx] = (row_idx, col_idx)

    return [coords[i] for i in range(len(bboxes))]


def match_descriptor_to_chamber(
    descriptor: ParsedDescriptor,
    grid_coords: Sequence[Tuple[int, int]],
) -> Optional[int]:
    """Resolve a ParsedDescriptor against a grid layout.

    Returns the bbox index that uniquely matches the descriptor, or None if
    zero or multiple chambers match.
    """
    if not grid_coords:
        return None

    n_rows = max(r for r, _ in grid_coords) + 1
    target_row: Optional[int] = None
    if descriptor.row_key == "top":
        target_row = descriptor.row_offset
    elif descriptor.row_key == "bottom":
        target_row = (n_rows - 1) - descriptor.row_offset
    elif descriptor.row_key == "middle":
        target_row = n_rows // 2

    if target_row is None or target_row < 0 or target_row >= n_rows:
        return None

    row_members = [(i, c) for i, (r, c) in enumerate(grid_coords) if r == target_row]
    if not row_members:
        return None

    if descriptor.col_key is None:
        # Need column hint to pick within a row of multiple chambers
        if len(row_members) == 1:
            return row_members[0][0]
        return None

    row_members.sort(key=lambda x: x[1])
    n_cols = len(row_members)

    if descriptor.col_key == "left":
        bucket = row_members[:max(1, n_cols // 3)]
    elif descriptor.col_key == "right":
        bucket = row_members[-max(1, n_cols // 3):]
    elif descriptor.col_key == "middle":
        if n_cols == 1:
            bucket = row_members
        elif n_cols == 2:
            return None  # ambiguous
        else:
            third = max(1, n_cols // 3)
            start = third
            end = n_cols - third
            if end <= start:
                start, end = n_cols // 2, n_cols // 2 + 1
            bucket = row_members[start:end]
    else:
        return None

    if len(bucket) == 1:
        return bucket[0][0]
    return None


def build_mapping(
    bboxes: Sequence[Tuple[int, int, int, int]],
    ref_chambers: Sequence[ReferenceChamber],
    override_path: Optional[Path] = None,
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """Compute descriptor -> detected chamber index for all reference chambers.

    Returns (mapping, ambiguities). `mapping[descriptor]` is the assigned bbox
    index or None if unresolved. `ambiguities` is the list of descriptors that
    the algorithm could not assign automatically. Overrides loaded from JSON
    win unconditionally.
    """
    overrides: Dict[str, int] = {}
    if override_path is not None:
        override_path = Path(override_path)
        if override_path.exists():
            try:
                overrides = {k: int(v) for k, v in json.loads(override_path.read_text()).items()
                             if v is not None}
            except (json.JSONDecodeError, ValueError, TypeError):
                overrides = {}

    grid_coords = assign_grid_coordinates(bboxes)
    used: set[int] = set()
    mapping: Dict[str, Optional[int]] = {}
    ambiguities: List[str] = []

    # First pass: apply overrides
    for chamber in ref_chambers:
        if chamber.descriptor in overrides:
            idx = overrides[chamber.descriptor]
            if 0 <= idx < len(bboxes):
                mapping[chamber.descriptor] = idx
                used.add(idx)

    # Second pass: auto-resolve remaining descriptors
    for chamber in ref_chambers:
        if chamber.descriptor in mapping:
            continue
        parsed = parse_descriptor(chamber.descriptor)
        idx = match_descriptor_to_chamber(parsed, grid_coords)
        if idx is not None and idx not in used:
            mapping[chamber.descriptor] = idx
            used.add(idx)
        else:
            mapping[chamber.descriptor] = None
            ambiguities.append(chamber.descriptor)

    return mapping, ambiguities


def save_mapping_override(
    mapping: Dict[str, Optional[int]],
    path: Path,
) -> None:
    """Persist a descriptor -> index mapping as JSON for next time."""
    payload = {k: v for k, v in mapping.items() if v is not None}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
