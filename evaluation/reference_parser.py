"""Parse the hand-curated `Sexy times (18+) ... .csv` ground-truth annotations.

The CSV layout is irregular: a stretch of blank padding rows, then a header row
containing the literal text `Chamber`, followed by chamber blocks. Each chamber
block starts with a row whose first cell is a positional descriptor like
`01 (bottom right)` and contains bout rows below it (chamber cell empty on
continuation rows). Blocks are separated by blank rows and may have footer
summary rows with only the `Duration` column populated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


_TIME_PATTERN = re.compile(r"^\s*(\d+):(\d{1,2})(?::(\d{1,2}))?\s*$")


def parse_time_string(value) -> Optional[float]:
    """Convert an `H:MM:SS` or `M:SS` time string to seconds.

    Returns None for blank/NaN/unparseable inputs. Strict about the colon
    structure so footer cells like a bare integer fall through as None."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    match = _TIME_PATTERN.match(text)
    if not match:
        return None
    a, b, c = match.groups()
    if c is None:
        return float(a) * 60.0 + float(b)
    return float(a) * 3600.0 + float(b) * 60.0 + float(c)


@dataclass
class ReferenceChamber:
    """One chamber's worth of annotated bouts."""
    descriptor: str
    bouts: List[Tuple[float, float]] = field(default_factory=list)
    source_rows: Tuple[int, int] = (0, 0)

    @property
    def total_seconds(self) -> float:
        return sum(end - start for start, end in self.bouts)


def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    """Locate the row whose first cell is the literal `Chamber`."""
    for idx in range(len(df)):
        cell = df.iat[idx, 0]
        if isinstance(cell, str) and cell.strip().lower() == "chamber":
            return idx
    return None


def parse_reference_csv(path: Path | str) -> List[ReferenceChamber]:
    """Read the reference CSV and return one ReferenceChamber per annotated arena."""
    path = Path(path)
    df = pd.read_csv(path, header=None, dtype=str, skip_blank_lines=False)
    if df.shape[1] < 3:
        return []

    header_idx = _find_header_row(df)
    if header_idx is None:
        return []

    chambers: List[ReferenceChamber] = []
    current: Optional[ReferenceChamber] = None

    for row_idx in range(header_idx + 1, len(df)):
        chamber_cell = df.iat[row_idx, 0]
        start_raw = df.iat[row_idx, 1] if df.shape[1] > 1 else None
        end_raw = df.iat[row_idx, 2] if df.shape[1] > 2 else None

        chamber_str = (str(chamber_cell).strip()
                       if isinstance(chamber_cell, str) and chamber_cell.strip()
                       and chamber_cell.strip().lower() != "nan"
                       else "")

        if chamber_str:
            if current is not None:
                current.source_rows = (current.source_rows[0], row_idx - 1)
                chambers.append(current)
            current = ReferenceChamber(descriptor=chamber_str, source_rows=(row_idx, row_idx))

        start_s = parse_time_string(start_raw)
        end_s = parse_time_string(end_raw)
        if current is not None and start_s is not None and end_s is not None and end_s > start_s:
            current.bouts.append((start_s, end_s))

    if current is not None:
        current.source_rows = (current.source_rows[0], len(df) - 1)
        chambers.append(current)

    return chambers
