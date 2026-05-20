"""Streamlit tab: verify chamber detection and map descriptors to detected boxes."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import streamlit as st

from utils.video import detect_chambers, find_best_detection_frame
from utils.overlay import draw_chamber_overlay

from .reference_parser import parse_reference_csv
from .spatial_mapper import (
    assign_grid_coordinates,
    build_mapping,
    parse_descriptor,
    save_mapping_override,
)


def _load_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    det = find_best_detection_frame(video_path)
    if det is not None:
        cap.release()
        return det
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def render_verify_tab():
    """Render the `Chamber Map` tab.

    Shows detected chambers on a representative frame and (if a reference CSV
    is supplied) maps each annotated descriptor to a detected chamber index,
    with a manual override path for any ambiguous match.
    """
    st.subheader("Chamber Detection + Reference Mapping")
    st.caption(
        "Visually confirm that every annotated chamber is detected and "
        "correctly aligned with its descriptor before running the comparison."
    )

    video_path = st.text_input(
        "Video path",
        placeholder="e.g. reference/movie.mp4",
        key="verify_video_path",
    )
    ref_csv_path = st.text_input(
        "Reference CSV path (optional)",
        placeholder="reference/Sexy times (18+)  - 4D CTRL GBX.csv",
        key="verify_ref_csv_path",
    )

    if not video_path or not Path(video_path).exists():
        st.info("Enter a valid video path to begin.")
        return

    cache_key = f"_vrf_{video_path}"
    if st.button("Detect chambers", key="verify_detect_btn") or cache_key not in st.session_state:
        frame = _load_frame(video_path)
        if frame is None:
            st.error("Could not read a frame from the video.")
            return
        bboxes = detect_chambers(frame, video_path=None)
        st.session_state[cache_key] = {"frame": frame, "bboxes": bboxes}

    state = st.session_state.get(cache_key)
    if not state:
        return

    frame, bboxes = state["frame"], state["bboxes"]

    ref_chambers = []
    mapping = {}
    ambig = []
    if ref_csv_path and Path(ref_csv_path).exists():
        try:
            ref_chambers = parse_reference_csv(ref_csv_path)
        except Exception as exc:
            st.error(f"Could not parse reference CSV: {exc}")
            ref_chambers = []
        if ref_chambers:
            override_path = Path(ref_csv_path).with_name("chamber_mapping.json")
            mapping, ambig = build_mapping(bboxes, ref_chambers, override_path)

    labels = [None] * len(bboxes)
    if mapping:
        inverse = {idx: desc for desc, idx in mapping.items() if idx is not None}
        for idx, desc in inverse.items():
            if 0 <= idx < len(labels):
                labels[idx] = desc

    overlay = draw_chamber_overlay(frame, bboxes, labels=labels)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Detected {len(bboxes)} chambers",
             use_container_width=True)

    if ref_chambers:
        st.markdown("### Descriptor → Detected Chamber")
        grid_coords = assign_grid_coordinates(bboxes)
        rows = []
        for chamber in ref_chambers:
            idx = mapping.get(chamber.descriptor)
            parsed = parse_descriptor(chamber.descriptor)
            rows.append({
                "Descriptor": chamber.descriptor,
                "Auto-assigned #": idx if idx is not None else "—",
                "Row anchor": f"{parsed.row_key} (+{parsed.row_offset})" if parsed.row_key else "—",
                "Col anchor": parsed.col_key or "—",
                "# Bouts": len(chamber.bouts),
                "Total (s)": f"{chamber.total_seconds:.1f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        if ambig:
            st.warning(
                f"{len(ambig)} descriptor(s) could not be auto-resolved. "
                "Use the manual override below to assign them."
            )

        st.markdown("### Manual override")
        st.caption(
            "Pick the detected chamber index for any descriptor that the auto "
            "mapper missed or got wrong. Saved overrides win on the next run."
        )
        override_path = Path(ref_csv_path).with_name("chamber_mapping.json")
        existing = {}
        if override_path.exists():
            try:
                existing = json.loads(override_path.read_text())
            except json.JSONDecodeError:
                existing = {}

        new_override = dict(existing)
        for chamber in ref_chambers:
            current = mapping.get(chamber.descriptor)
            default_idx = current if current is not None else 0
            options = list(range(len(bboxes)))
            label = chamber.descriptor
            choice = st.selectbox(
                label,
                options=[None] + options,
                index=options.index(default_idx) + 1 if current is not None else 0,
                format_func=lambda v: "— unassigned —" if v is None else f"Chamber #{v}",
                key=f"verify_override_{label}",
            )
            if choice is not None:
                new_override[chamber.descriptor] = int(choice)
            elif chamber.descriptor in new_override:
                del new_override[chamber.descriptor]

        if st.button("Save mapping", key="verify_save_mapping_btn"):
            save_mapping_override({k: v for k, v in new_override.items()},
                                  override_path)
            st.success(f"Wrote {override_path.name}")
