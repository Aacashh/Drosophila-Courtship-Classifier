"""Streamlit tab: compare predicted courtship bouts to reference annotations and tune."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from classification.heuristic import DEFAULT_PARAMS

from .calibration import estimate_px_per_mm
from .compare import evaluate_chamber, evaluate_video, plot_timeline
from .reference_parser import parse_reference_csv
from .spatial_mapper import build_mapping
from .tuning import (
    SLIDER_PARAMS,
    build_cache,
    courtship_bouts_seconds,
    diff_from_default,
    load_tracks,
    load_tuned_params,
    rescore,
    save_tuned_params,
    tracks_npz_path,
)


def _format_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


def render_compare_tab():
    """Render the `Compare to Reference` tab."""
    st.subheader("Compare predictions to ground truth + tune heuristics")
    st.caption(
        "Loads a previous Analyze-tab run, scores each mapped chamber against "
        "the reference CSV, and lets you sweep the dominant thresholds in real "
        "time using cached features."
    )

    results_dir_str = st.text_input(
        "Results directory",
        placeholder="analysis_results/2026-...",
        key="cmp_results_dir",
    )
    ref_csv_str = st.text_input(
        "Reference CSV path",
        placeholder="reference/Sexy times (18+)  - 4D CTRL GBX.csv",
        key="cmp_ref_csv",
    )

    if not results_dir_str or not Path(results_dir_str).exists():
        st.info("Enter a path to a results directory containing chamber_N_results.csv.")
        return
    if not ref_csv_str or not Path(ref_csv_str).exists():
        st.info("Enter the reference CSV path.")
        return

    results_dir = Path(results_dir_str)
    ref_csv = Path(ref_csv_str)

    ref_chambers = parse_reference_csv(ref_csv)
    if not ref_chambers:
        st.error("Reference CSV produced zero chambers — check the file format.")
        return

    # Build mapping. The Compare tab reuses any chamber_mapping.json saved by
    # the Verify tab.
    override_path = ref_csv.with_name("chamber_mapping.json")
    overrides: Dict[str, int] = {}
    if override_path.exists():
        try:
            overrides = {k: int(v) for k, v in json.loads(override_path.read_text()).items()
                         if v is not None}
        except (json.JSONDecodeError, ValueError):
            overrides = {}

    if not overrides:
        st.warning(
            "No `chamber_mapping.json` found next to the reference CSV. "
            "Run the Chamber Map tab first to establish the mapping, or rely on "
            "the auto-mapper from any cached chamber bboxes. For now the Compare "
            "tab requires the saved mapping."
        )
        return

    mapping: Dict[str, Optional[int]] = {desc: overrides.get(desc) for desc in
                                         [c.descriptor for c in ref_chambers]}

    # Determine fps + n_frames from the first available chamber's tracks NPZ.
    fps = 30.0
    n_frames = 0
    for chamber in ref_chambers:
        idx = mapping.get(chamber.descriptor)
        if idx is None:
            continue
        loaded = load_tracks(tracks_npz_path(results_dir, idx))
        if loaded is not None:
            tracks_sample, fps_sample = loaded
            fps = fps_sample
            n_frames = len(tracks_sample["x"][0]) if tracks_sample.get("x") else 0
            break

    if n_frames == 0:
        st.error(
            "No tracks NPZ found in the results directory. Re-run the Analyze "
            "tab so that chamber_N_tracks.npz files are written."
        )
        return

    duration_s = n_frames / fps if fps > 0 else 600.0
    st.write(f"Detected fps={fps:.1f}, n_frames={n_frames} ({_format_seconds(duration_s)}).")

    # ------------------------------------------------------------------
    # Build feature caches lazily for every mapped chamber.
    # ------------------------------------------------------------------
    cache_key = f"_cmp_cache::{results_dir}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}
    feature_cache = st.session_state[cache_key]

    # Auto-calibrate px_per_mm from the first chamber's tracks
    auto_scale_key = f"_cmp_auto_scale::{results_dir}"
    if auto_scale_key not in st.session_state:
        scale_samples = []
        for chamber in ref_chambers:
            idx = mapping.get(chamber.descriptor)
            if idx is None:
                continue
            loaded = load_tracks(tracks_npz_path(results_dir, idx))
            if loaded is None:
                continue
            tracks, _ = loaded
            est = estimate_px_per_mm(tracks)
            if est is not None:
                scale_samples.append(est)
        if scale_samples:
            st.session_state[auto_scale_key] = float(sum(scale_samples) / len(scale_samples))
        else:
            st.session_state[auto_scale_key] = 15.0

    st.markdown("### Calibration")
    body_length_mm = st.number_input(
        "Assumed fly body length (mm)",
        value=2.5, min_value=1.0, max_value=4.0, step=0.1,
        key="cmp_body_length",
    )

    if st.button("Re-estimate px/mm with new body length", key="cmp_recalibrate"):
        scale_samples = []
        for chamber in ref_chambers:
            idx = mapping.get(chamber.descriptor)
            if idx is None:
                continue
            loaded = load_tracks(tracks_npz_path(results_dir, idx))
            if loaded is None:
                continue
            tracks, _ = loaded
            est = estimate_px_per_mm(tracks, assumed_body_length_mm=body_length_mm)
            if est is not None:
                scale_samples.append(est)
        if scale_samples:
            st.session_state[auto_scale_key] = float(sum(scale_samples) / len(scale_samples))

    px_per_mm = st.number_input(
        "px / mm (override if needed)",
        value=float(st.session_state[auto_scale_key]),
        min_value=1.0, max_value=200.0, step=0.5,
        key="cmp_px_per_mm",
    )

    # ------------------------------------------------------------------
    # Sliders for the dominant thresholds.
    # ------------------------------------------------------------------
    st.markdown("### Heuristic thresholds")
    tuned_path = results_dir / "tuned_params.json"
    base_params = load_tuned_params(tuned_path)

    params: Dict[str, float] = dict(base_params)
    cols = st.columns(2)
    for i, (name, lo, hi, step) in enumerate(SLIDER_PARAMS):
        with cols[i % 2]:
            params[name] = st.slider(
                name,
                min_value=float(lo),
                max_value=float(hi),
                value=float(base_params.get(name, DEFAULT_PARAMS.get(name, lo))),
                step=float(step),
                key=f"cmp_slider_{name}",
            )

    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("Save these params to tuned_params.json", key="cmp_save_params"):
            save_tuned_params(params, tuned_path)
            st.success(f"Saved {tuned_path.name} ({len(diff_from_default(params))} overrides)")
    with reset_col:
        if st.button("Reset to defaults", key="cmp_reset_params"):
            for name, _, _, _ in SLIDER_PARAMS:
                key = f"cmp_slider_{name}"
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ------------------------------------------------------------------
    # Score every mapped chamber with current sliders.
    # ------------------------------------------------------------------
    st.markdown("### Per-chamber metrics")
    pred_by_desc: Dict[str, List] = {}
    ref_by_desc: Dict[str, List] = {}
    metric_rows = []
    for chamber in ref_chambers:
        ref_by_desc[chamber.descriptor] = chamber.bouts
        idx = mapping.get(chamber.descriptor)
        if idx is None:
            metric_rows.append({
                "descriptor": chamber.descriptor,
                "chamber": "—",
                "scope": "unmapped",
            })
            pred_by_desc[chamber.descriptor] = []
            continue

        if idx not in feature_cache:
            loaded = load_tracks(tracks_npz_path(results_dir, idx))
            if loaded is None:
                metric_rows.append({
                    "descriptor": chamber.descriptor,
                    "chamber": idx,
                    "scope": "no_tracks",
                })
                pred_by_desc[chamber.descriptor] = []
                continue
            tracks, fps_local = loaded
            feature_cache[idx] = build_cache(idx, tracks, fps_local, px_per_mm)

        cached = feature_cache[idx]
        bouts = rescore(cached, params=params, px_per_mm_override=px_per_mm)
        pred_seconds, scope = courtship_bouts_seconds(
            bouts, cached.fps, male_only=True, tracks=cached.tracks
        )
        pred_by_desc[chamber.descriptor] = pred_seconds

        n_local = cached.features["n_frames"]
        metrics = evaluate_chamber(pred_seconds, chamber.bouts,
                                   n_frames=n_local, fps=cached.fps)
        metric_rows.append({
            "descriptor": chamber.descriptor,
            "chamber": idx,
            "scope": scope,
            "F1": round(metrics["frame_f1"], 3),
            "Precision": round(metrics["frame_precision"], 3),
            "Recall": round(metrics["frame_recall"], 3),
            "IoU": round(metrics["frame_iou"], 3),
            "Bouts matched": metrics["bout_matched"],
            "Bouts missed": metrics["bout_missed"],
            "Bouts FP": metrics["bout_fp"],
            "Pred dur (s)": round(metrics["pred_total_s"], 1),
            "Ref dur (s)": round(metrics["ref_total_s"], 1),
        })

    st.dataframe(metric_rows, use_container_width=True, hide_index=True)

    # Aggregate F1 across resolved chambers
    f1s = [r["F1"] for r in metric_rows if "F1" in r]
    if f1s:
        st.metric("Mean F1 across chambers", f"{sum(f1s) / len(f1s):.3f}")

    # ------------------------------------------------------------------
    # Timeline visualization
    # ------------------------------------------------------------------
    st.markdown("### Timeline (red = reference, blue = predicted)")
    if pred_by_desc:
        fig = plot_timeline(pred_by_desc, ref_by_desc, duration_s=duration_s)
        st.pyplot(fig, clear_figure=True)

        if st.button("Save timeline PNG", key="cmp_save_timeline"):
            png_path = results_dir / "evaluation_timeline.png"
            fig.savefig(png_path, dpi=120)
            st.success(f"Saved {png_path.name}")
