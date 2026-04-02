import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import datetime
import json
import gc
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from PIL import Image

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from utils.video import detect_chambers, crop_video, stabilize_video
from tracking.tracker import track_two_flies
from tracking.features import build_feature_matrix
from classification.jab_parser import load_jab_models
from classification.inference import eval_boosting, scores_to_bouts
from classification.heuristic import HeuristicClassifier, DEFAULT_PARAMS
from utils.export import export_to_csv, infer_sex, export_summary_csv
from classification.heuristic import compute_courtship_index

st.set_page_config(page_title="Courtship Analysis", layout="wide")


# --- Navigation callbacks (must be at module level for on_click) ---

def _vv_prev_bout_sel():
    idx = st.session_state.get('vv_bout_sel', 0)
    if idx > 0:
        st.session_state.vv_bout_sel = idx - 1


def _vv_next_bout_sel():
    idx = st.session_state.get('vv_bout_sel', 0)
    total = st.session_state.get('_vv_bout_count', 1)
    if idx < total - 1:
        st.session_state.vv_bout_sel = idx + 1


@st.cache_data
def _generate_bout_gif(video_path_str, start_frame, end_frame, target_fps=10, max_width=400, max_frames=100):
    """Generate a lightweight GIF preview of a behavior bout."""
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return None
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(vid_fps / target_fps))

    pil_frames = []
    for i in range(start_frame, end_frame + 1, step):
        if len(pil_frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > max_width:
            scale = max_width / w
            rgb = cv2.resize(rgb, (max_width, int(h * scale)))
        pil_frames.append(Image.fromarray(rgb))
    cap.release()

    if not pil_frames:
        return None
    buf = io.BytesIO()
    duration_ms = max(50, int(1000 / target_fps))
    pil_frames[0].save(buf, format='GIF', save_all=True, append_images=pil_frames[1:],
                       duration=duration_ms, loop=0)
    return buf.getvalue()


def process_single_chamber(i, roi, video_path, output_dir, do_stabilize, analysis_mode, models, px_per_mm, heuristic_params=None):
    """Worker function for parallel processing."""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Crop
        chamber_video = output_dir / f"chamber_{i}.mp4"
        crop_video(Path(video_path), chamber_video, roi)

        if not chamber_video.exists() or chamber_video.stat().st_size == 0:
            return i, False, f"Crop failed - output video empty or missing", []
        
        # 2. Stabilize
        final_video = chamber_video
        if do_stabilize:
            stab_video = output_dir / f"chamber_{i}_stab.mp4"
            ok = stabilize_video(str(chamber_video), str(stab_video))
            if ok:
                final_video = stab_video
        
        # 3. Track
        tracks = track_two_flies(final_video)
        
        # Get FPS
        cap_temp = cv2.VideoCapture(str(final_video))
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
        cap_temp.release()
        
        # 4. Classify
        bouts_by_behavior = {}
        n_flies = 2
        
        if analysis_mode == "JAABA Classifiers":
            for model in models:
                bouts_by_behavior[model.behavior] = []
                for fly in range(n_flies):
                    X = build_feature_matrix(tracks, model.feature_names, fly)
                    scores = eval_boosting(model.learners, X)
                    bouts = scores_to_bouts(scores, threshold=0.0, min_len=3)
                    bouts_by_behavior[model.behavior].append(bouts)
        else:
            # Heuristic
            bouts_by_behavior = HeuristicClassifier.classify(tracks, fps, px_per_mm, params=heuristic_params)
        
        # 5. Export
        fly_stats = infer_sex(bouts_by_behavior)
        csv_path = output_dir / f"chamber_{i}_results.csv"
        export_to_csv(csv_path, bouts_by_behavior, fps, fly_stats)

        # Export summary CSV with CI and latency
        n_frames = len(tracks['x'][0])
        summary_path = output_dir / f"chamber_{i}_summary.csv"
        export_summary_csv(summary_path, bouts_by_behavior, fps, n_frames, fly_stats)

        # Return summary for visualization (includes CI)
        summary = []
        for fly_idx in range(n_flies):
            ci = compute_courtship_index(bouts_by_behavior, n_frames, fly_idx)
            fly_label = f"Fly {fly_idx}"
            if fly_idx in fly_stats:
                if fly_stats[fly_idx].get('is_male'):
                    fly_label += " (Male)"
                elif fly_stats[fly_idx].get('is_female'):
                    fly_label += " (Female)"
            summary.append({
                'Chamber': i,
                'Fly': fly_label,
                'Behavior': 'Courtship Index',
                'Count': '',
                'Total Duration (s)': f"{ci:.1f}%"
            })
        for beh, bouts_list in bouts_by_behavior.items():
            for fly_idx, bouts in enumerate(bouts_list):
                duration = sum((e - s + 1) for s, e in bouts) / fps
                fly_label = f"Fly {fly_idx}"
                if fly_idx in fly_stats:
                    if fly_stats[fly_idx].get('is_male'):
                        fly_label += " (Male)"
                    elif fly_stats[fly_idx].get('is_female'):
                        fly_label += " (Female)"
                summary.append({
                    'Chamber': i,
                    'Fly': fly_label,
                    'Behavior': beh,
                    'Count': len(bouts),
                    'Total Duration (s)': round(duration, 2)
                })

        # Cleanup large objects to free memory for next chamber
        del tracks, bouts_by_behavior
        gc.collect()

        return i, True, csv_path, summary
    except Exception as e:
        gc.collect()
        return i, False, str(e), []


def _discover_result_dirs():
    """Find all result directories. Returns list of (path_str, display_label) sorted newest first."""
    found = []
    base = Path("analysis_results")
    if base.exists():
        # Check timestamped subdirectories
        for sub in sorted(base.iterdir(), reverse=True):
            if sub.is_dir() and list(sub.glob("*_results.csv")):
                # Parse timestamp from folder name for display
                try:
                    dt = datetime.datetime.strptime(sub.name, "%Y-%m-%d_%H-%M-%S")
                    label = dt.strftime("%b %d, %Y %I:%M %p")
                except ValueError:
                    label = sub.name
                found.append((str(sub), f"{label}  ({sub.name})"))
        # Also check flat analysis_results/ (legacy)
        if list(base.glob("*_results.csv")):
            found.append((str(base), "analysis_results (legacy, no timestamp)"))

    # Legacy courtship_results/
    legacy = Path("courtship_results")
    if legacy.exists() and list(legacy.glob("*_results.csv")):
        found.append((str(legacy), "courtship_results (legacy)"))

    # Session state from current run
    sess_dir = st.session_state.get('_analysis_output_dir')
    if sess_dir:
        sess_path = Path(sess_dir)
        if sess_path.exists() and list(sess_path.glob("*_results.csv")):
            if not any(p == sess_dir for p, _ in found):
                found.insert(0, (sess_dir, f"Current session ({sess_path.name})"))

    return found


def verification_viewer():
    """Visual verification tool for reviewing chamber analysis results."""

    # Sidebar controls for verification
    st.sidebar.markdown("---")
    st.sidebar.subheader("Verification Settings")
    max_preview_frames = st.sidebar.slider("Preview Max Frames", 10, 300, 100, key="vv_max_frames")

    # Auto-detect results directories (timestamped, newest first)
    result_dirs = _discover_result_dirs()

    if not result_dirs:
        st.info("No analysis results found. Run analysis first.")
        return

    # --- Top controls: directory, chamber, male filter ---
    top_cols = st.columns([2, 2, 2])
    with top_cols[0]:
        dir_labels = [label for _, label in result_dirs]
        dir_paths = [p for p, _ in result_dirs]
        sel_idx_dir = st.selectbox("Results Folder", range(len(dir_labels)),
                                   format_func=lambda i: dir_labels[i], key="vv_dir")
        results_dir = dir_paths[sel_idx_dir]
    results_path = Path(results_dir)

    csv_files = sorted(results_path.glob("*_results.csv"))
    chambers = []
    for f in csv_files:
        try:
            num = int(f.stem.replace("chamber_", "").replace("_results", ""))
            chambers.append(num)
        except ValueError:
            continue

    if not chambers:
        st.info("No chamber results found.")
        return
    chambers.sort()

    with top_cols[1]:
        selected = st.selectbox("Chamber", chambers, format_func=lambda x: f"Chamber #{x}", key="vv_chamber")
    with top_cols[2]:
        male_only = st.checkbox("Male Only", value=False, key="vv_male_only")

    # Load CSV
    csv_path = results_path / f"chamber_{selected}_results.csv"
    df_raw = pd.read_csv(csv_path)

    if 'Start_Frame' not in df_raw.columns or 'End_Frame' not in df_raw.columns:
        st.error("Results CSV missing Start_Frame/End_Frame columns.")
        return

    # Apply male filter
    if male_only:
        has_male = df_raw['Fly'].str.contains('Male', case=False, na=False).any()
        if has_male:
            df = df_raw[df_raw['Fly'].str.contains('Male', case=False, na=False)].copy()
        else:
            st.warning("No fly identified as Male in this chamber. Showing all data.")
            df = df_raw.copy()
    else:
        df = df_raw.copy()

    # Find video (prefer stabilized)
    video_path = results_path / f"chamber_{selected}_stab.mp4"
    if not video_path.exists():
        video_path = results_path / f"chamber_{selected}.mp4"
    has_video = video_path.exists()

    # Get video info
    fps = 30.0
    total_frames = 0
    if has_video:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    # --- Cumulative courtship duration metric ---
    if not df.empty and 'Duration (s)' in df.columns:
        total_dur = df['Duration (s)'].astype(float).sum()
        fly_label = "Male" if male_only else "All Flies"
        st.metric(f"Total Courtship Duration ({fly_label})", f"{total_dur:.1f}s")

    # --- Bout List with Auto GIF Preview ---
    bout_data = []
    for _, row in df.iterrows():
        sf = int(row['Start_Frame'])
        ef = int(row['End_Frame'])
        t_s = sf / fps
        dur = float(row.get('Duration (s)', (ef - sf + 1) / fps))
        label = f"{row['Behavior']} | {row['Fly']} | {t_s:.1f}s ({dur:.2f}s)"
        bout_data.append({'label': label, 'start': sf, 'end': ef,
                          'behavior': row['Behavior'], 'fly': str(row['Fly'])})

    st.session_state._vv_bout_count = len(bout_data)
    sel_idx = 0

    if bout_data:
        col_sel, col_prev, col_next = st.columns([4, 1, 1])
        with col_sel:
            sel_idx = st.selectbox("Select Bout", range(len(bout_data)),
                                   format_func=lambda i: bout_data[i]['label'], key="vv_bout_sel")
        with col_prev:
            st.write("")
            st.write("")
            st.button("Prev Bout", on_click=_vv_prev_bout_sel, key="vv_pbout")
        with col_next:
            st.write("")
            st.write("")
            st.button("Next Bout", on_click=_vv_next_bout_sel, key="vv_nbout")

        # Auto-show GIF preview for selected bout
        if has_video and sel_idx < len(bout_data):
            sel_bout = bout_data[sel_idx]
            with st.spinner("Generating preview..."):
                gif_bytes = _generate_bout_gif(str(video_path), sel_bout['start'], sel_bout['end'],
                                               max_frames=max_preview_frames)
            if gif_bytes:
                st.image(gif_bytes,
                         caption=f"{sel_bout['behavior']} | {sel_bout['fly']} | "
                                 f"{sel_bout['start']/fps:.1f}s - {sel_bout['end']/fps:.1f}s")
            else:
                st.warning("Could not generate preview for this bout.")
        elif not has_video:
            st.caption(f"No video available for Chamber #{selected} — run analysis to generate previews.")
    else:
        st.info("No bouts detected in this chamber.")

    # --- Behavior Timeline ---
    if not df.empty:
        st.write("### Behavior Timeline")

        behaviors = list(df['Behavior'].unique())
        colors_map = {
            'WingExt': '#FF6B6B',
            'Following': '#4ECDC4',
            'Circling': '#45B7D1',
            'Copulation': '#96CEB4',
            'Attempted_Copulation': '#FFEAA7'
        }

        if behaviors:
            fig, ax = plt.subplots(figsize=(14, max(1.5, len(behaviors) * 0.8 + 0.5)))

            for _, row in df.iterrows():
                beh = row['Behavior']
                beh_idx = behaviors.index(beh)
                start_s = int(row['Start_Frame']) / fps
                end_s = int(row['End_Frame']) / fps
                dur = max(end_s - start_s, 0.5 / fps)
                color = colors_map.get(beh, '#999999')

                fly_str = str(row['Fly'])
                if '1' in fly_str.split('Fly')[-1][:3]:
                    y_pos = beh_idx + 0.15
                    alpha = 0.5
                else:
                    y_pos = beh_idx - 0.15
                    alpha = 0.85

                ax.barh(y_pos, dur, left=start_s, height=0.25, color=color, alpha=alpha,
                        edgecolor='black', linewidth=0.3)

            # Highlight selected bout on timeline
            if bout_data and sel_idx < len(bout_data):
                sel_bout = bout_data[sel_idx]
                sel_start_s = sel_bout['start'] / fps
                sel_end_s = sel_bout['end'] / fps
                ax.axvspan(sel_start_s, sel_end_s, alpha=0.2, color='red')

            ax.set_yticks(range(len(behaviors)))
            ax.set_yticklabels(behaviors, fontsize=9)
            ax.set_xlabel("Time (s)")
            video_dur = total_frames / fps if total_frames > 0 else (int(df['End_Frame'].max()) / fps)
            ax.set_xlim(0, video_dur)
            ax.set_title(f"Chamber #{selected} — Detected Behavior Bouts")

            patches = [mpatches.Patch(color=colors_map.get(b, '#999'), label=b) for b in behaviors]
            patches.append(mpatches.Patch(color='red', alpha=0.2, label='Selected Bout'))
            ax.legend(handles=patches, loc='upper right', fontsize=7)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Summary CSV
    summary_path = results_path / f"chamber_{selected}_summary.csv"
    if summary_path.exists():
        with st.expander("Summary Statistics"):
            summary_df = pd.read_csv(summary_path)
            if male_only and 'Fly' in summary_df.columns:
                has_male_s = summary_df['Fly'].str.contains('Male', case=False, na=False).any()
                if has_male_s:
                    summary_df = summary_df[summary_df['Fly'].str.contains('Male', case=False, na=False)]
            st.dataframe(summary_df, use_container_width=True)

    with st.expander("Full Results Table"):
        st.dataframe(df, use_container_width=True)


def _run_analysis():
    """Main analysis workflow."""
    # --- Step 1: Configuration & Inputs ---
    st.sidebar.header("Settings")
    
    analysis_mode = st.sidebar.selectbox("Analysis Method", ["Heuristic (Built-in Rules)", "JAABA Classifiers"])
    
    models = []
    if analysis_mode == "JAABA Classifiers":
        jab_source = st.sidebar.radio("Classifiers Source", ["Default Folder", "Upload .jab Files"])
        jab_dir = Path("Courtship_classifiers") # Default
        
        uploaded_jabs = []
        if jab_source == "Upload .jab Files":
            uploaded_jabs = st.sidebar.file_uploader("Upload .jab files", accept_multiple_files=True, type=["jab"])
    
    st.sidebar.subheader("Parameters")
    px_per_mm = st.sidebar.number_input("Pixels per mm (Calibration)", value=15.0, min_value=1.0)

    # --- Heuristic Threshold Controls ---
    heuristic_params = None
    if analysis_mode == "Heuristic (Built-in Rules)":
        with st.sidebar.expander("Heuristic Thresholds"):
            st.caption("Adjust classification thresholds. Defaults follow literature standards.")
            d = DEFAULT_PARAMS

            st.markdown("**Wing Extension**")
            h_wing_angle = st.number_input("Angle (deg)", value=d['wing_ext_angle_deg'], min_value=10, max_value=120, key="h_wing_angle")
            h_wing_dur = st.number_input("Min Duration (s)", value=d['wing_ext_min_dur_s'], min_value=0.1, max_value=10.0, step=0.1, key="h_wing_dur")

            st.markdown("**Following**")
            h_fol_vel = st.number_input("Forward Velocity (mm/s)", value=d['follow_velocity_mm_s'], min_value=0.1, max_value=20.0, step=0.1, key="h_fol_vel")
            h_fol_facing = st.number_input("Facing Angle (deg)", value=d['follow_facing_deg'], min_value=10, max_value=90, key="h_fol_facing")
            h_fol_dur = st.number_input("Min Duration (s)", value=d['follow_min_dur_s'], min_value=0.1, max_value=10.0, step=0.1, key="h_fol_dur")

            st.markdown("**Circling**")
            h_cir_vel = st.number_input("Lateral Velocity (mm/s)", value=d['circling_lat_vel_mm_s'], min_value=0.1, max_value=20.0, step=0.1, key="h_cir_vel")
            h_cir_dur = st.number_input("Min Duration (s)", value=d['circling_min_dur_s'], min_value=0.1, max_value=10.0, step=0.1, key="h_cir_dur")

            st.markdown("**Copulation**")
            h_cop_dist = st.number_input("Max Distance (mm)", value=d['cop_distance_mm'], min_value=0.1, max_value=10.0, step=0.1, key="h_cop_dist")
            h_cop_vel = st.number_input("Max Velocity (mm/s)", value=d['cop_velocity_mm_s'], min_value=0.01, max_value=5.0, step=0.05, key="h_cop_vel")
            h_cop_dur = st.number_input("Min Duration (s)", value=d['cop_min_dur_s'], min_value=1.0, max_value=60.0, step=1.0, key="h_cop_dur")

            st.markdown("**Attempted Copulation**")
            h_att_dist = st.number_input("Nose Distance (mm)", value=d['att_cop_nose_dist_mm'], min_value=0.1, max_value=5.0, step=0.1, key="h_att_dist")
            h_att_dur = st.number_input("Min Duration (s)", value=d['att_cop_min_dur_s'], min_value=0.1, max_value=10.0, step=0.1, key="h_att_dur")

            st.markdown("**General**")
            h_merge = st.number_input("Merge Gap (s)", value=d['merge_gap_s'], min_value=0.0, max_value=5.0, step=0.1, key="h_merge")

        heuristic_params = {
            'wing_ext_angle_deg': h_wing_angle,
            'wing_ext_min_dur_s': h_wing_dur,
            'follow_velocity_mm_s': h_fol_vel,
            'follow_facing_deg': h_fol_facing,
            'follow_min_dur_s': h_fol_dur,
            'circling_lat_vel_mm_s': h_cir_vel,
            'circling_min_dur_s': h_cir_dur,
            'cop_distance_mm': h_cop_dist,
            'cop_velocity_mm_s': h_cop_vel,
            'cop_min_dur_s': h_cop_dur,
            'att_cop_nose_dist_mm': h_att_dist,
            'att_cop_min_dur_s': h_att_dur,
            'merge_gap_s': h_merge,
        }

    # --- Main Workflow ---

    video_source = st.radio("Video Source", ["Upload File", "Local File Path"], horizontal=True)

    video_path = None
    if video_source == "Upload File":
        uploaded_video = st.file_uploader("Choose a Video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()
            video_path = tfile.name
    else:
        path_input = st.text_input("Enter full path to video file",
                                   placeholder="e.g. D:/Videos/experiment.mp4")
        if path_input:
            if Path(path_input).is_file():
                video_path = path_input
            else:
                st.error(f"File not found: {path_input}")

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            st.error("Could not read video.")
            return
            
        st.subheader("1. Chamber Detection")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # ROI Editing state
            if 'rois' not in st.session_state:
                st.session_state.rois = detect_chambers(first_frame, video_path=video_path)
                
            # --- Interactive Grid Adjustment ---
            st.write("### Fine-tune Detection")
            if st.button("Re-run Auto Detection"):
                st.session_state.rois = detect_chambers(first_frame, video_path=video_path)
                st.rerun()

            # Display current ROIs
            frame_viz = first_frame.copy()
            for i, (x, y, w, h) in enumerate(st.session_state.rois):
                cv2.rectangle(frame_viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_viz, f"#{i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            st.image(cv2.cvtColor(frame_viz, cv2.COLOR_BGR2RGB), caption="Detected Chambers", use_container_width=True)
            
        with col2:
            st.write(f"Detected Chambers: {len(st.session_state.rois)}")
            
            # --- Manual Grid Builder ---
            st.write("### Manual Grid Builder")
            mg_cols = st.columns(2)
            with mg_cols[0]:
                n_rows = st.number_input("Rows", 1, 20, 4)
                start_x = st.number_input("Start X", 0, first_frame.shape[1], 50)
                box_w = st.number_input("Box Width", 10, 500, 100)
                gap_x = st.number_input("Gap X", 0, 100, 10)
            with mg_cols[1]:
                n_cols = st.number_input("Cols", 1, 20, 6)
                start_y = st.number_input("Start Y", 0, first_frame.shape[0], 50)
                box_h = st.number_input("Box Height", 10, 500, 100)
                gap_y = st.number_input("Gap Y", 0, 100, 10)
                
            if st.button("Generate Grid"):
                new_rois = []
                for r in range(n_rows):
                    for c in range(n_cols):
                        rx = start_x + c * (box_w + gap_x)
                        ry = start_y + r * (box_h + gap_y)
                        if rx + box_w <= first_frame.shape[1] and ry + box_h <= first_frame.shape[0]:
                            new_rois.append((rx, ry, box_w, box_h))
                st.session_state.rois = new_rois
                st.rerun()

            st.write("---")
            st.write("Manual ROI Override (Raw)")
            roi_input = st.text_area("Enter ROIs as lines of x,y,w,h", 
                                     value="\n".join([f"{r[0]},{r[1]},{r[2]},{r[3]}" for r in st.session_state.rois]))
            
            if st.button("Update ROIs"):
                try:
                    new_rois = []
                    for line in roi_input.strip().split('\n'):
                        if line.strip():
                            parts = [int(p.strip()) for p in line.split(',')]
                            if len(parts) == 4:
                                new_rois.append(tuple(parts))
                    st.session_state.rois = new_rois
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid format: {e}")

        # --- Step 2: Processing ---
        st.subheader("2. Analysis")
        do_stabilize = st.checkbox("Enable Stabilization (Slower)", value=True)
        
        # Chamber Selection
        st.write("### Select Chambers to Process")
        
        # Initialize selection state if not present
        if 'chamber_selection' not in st.session_state or len(st.session_state.chamber_selection) != len(st.session_state.rois):
            st.session_state.chamber_selection = [True] * len(st.session_state.rois)

        # Select/Deselect All Buttons
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            if st.button("Select All"):
                st.session_state.chamber_selection = [True] * len(st.session_state.rois)
                st.rerun()
        with sel_col2:
            if st.button("Deselect All"):
                st.session_state.chamber_selection = [False] * len(st.session_state.rois)
                st.rerun()

        # Create a grid of checkboxes
        selected_indices = []
        cols = st.columns(4)
        for i in range(len(st.session_state.rois)):
            with cols[i % 4]:
                # Update session state based on checkbox interaction
                new_val = st.checkbox(f"Chamber #{i}", value=st.session_state.chamber_selection[i], key=f"chk_{i}")
                st.session_state.chamber_selection[i] = new_val
                if new_val:
                    selected_indices.append(i)
        
        if st.button("Start Analysis", type="primary"):
            if not selected_indices:
                st.error("No chambers selected. Please select at least one chamber to proceed.")
                return
                
            # Prepare Classifiers if JAABA
            if analysis_mode == "JAABA Classifiers":
                if jab_source == "Upload .jab Files" and uploaded_jabs:
                    jab_tmp = Path(tempfile.mkdtemp())
                    for uf in uploaded_jabs:
                        (jab_tmp / uf.name).write_bytes(uf.getvalue())
                    models = load_jab_models(jab_tmp)
                else:
                    if not jab_dir.exists():
                        st.error(f"Default classifier directory {jab_dir} not found. Please upload files.")
                        return
                    models = load_jab_models(jab_dir)
                    
                if not models:
                    st.error("No classifiers found!")
                    return
                st.success(f"Loaded {len(models)} classifiers: {[m.behavior for m in models]}")
            
            # Create timestamped output dir
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = Path("analysis_results") / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing...")
            
            # Visualization containers
            st.subheader("3. Real-time Results")
            results_container = st.container()
            
            # --- RAM-aware worker count ---
            if _HAS_PSUTIL:
                avail_gb = psutil.virtual_memory().available / (1024 ** 3)
                max_by_ram = max(1, int(avail_gb / 0.5))
                max_workers = min(max_by_ram, len(selected_indices), 4)
            else:
                max_workers = min(2, len(selected_indices))
            status_text.text(f"Using {max_workers} parallel worker(s)...")

            # --- Resume: skip already-completed chambers ---
            progress_file = output_dir / "_progress.json"
            completed_chambers = set()
            if progress_file.exists():
                try:
                    completed_chambers = set(json.loads(progress_file.read_text()))
                except Exception:
                    completed_chambers = set()

            all_summaries = []
            chambers_to_run = []
            for i in selected_indices:
                existing_csv = output_dir / f"chamber_{i}_results.csv"
                if i in completed_chambers and existing_csv.exists():
                    # Load existing results for display
                    try:
                        df_existing = pd.read_csv(existing_csv)
                        for _, row in df_existing.iterrows():
                            all_summaries.append({
                                'Chamber': i, 'Fly': row.get('Fly', ''),
                                'Behavior': row.get('Behavior', ''),
                                'Count': 1, 'Total Duration (s)': row.get('Duration (s)', 0)
                            })
                    except Exception:
                        pass
                    status_text.text(f"Chamber {i} already complete — skipping")
                else:
                    chambers_to_run.append(i)

            if chambers_to_run:
                skipped = len(selected_indices) - len(chambers_to_run)
                if skipped > 0:
                    st.info(f"Resuming: {skipped} chamber(s) already done, {len(chambers_to_run)} remaining")

            total_to_process = len(selected_indices)
            completed = total_to_process - len(chambers_to_run)
            if completed > 0:
                progress_bar.progress(completed / total_to_process)

            CHAMBER_TIMEOUT = 600  # 10 minutes per chamber

            if chambers_to_run:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {}
                    for i in chambers_to_run:
                        roi = st.session_state.rois[i]
                        f = executor.submit(
                            process_single_chamber, i, roi, video_path,
                            output_dir, do_stabilize, analysis_mode, models,
                            px_per_mm, heuristic_params
                        )
                        future_to_idx[f] = i

                    for future in concurrent.futures.as_completed(future_to_idx):
                        chamber_idx = future_to_idx[future]
                        try:
                            i, success, result_path, summary = future.result(timeout=CHAMBER_TIMEOUT)
                        except concurrent.futures.TimeoutError:
                            st.warning(f"Chamber {chamber_idx} timed out after {CHAMBER_TIMEOUT}s — skipping")
                            completed += 1
                            progress_bar.progress(completed / total_to_process)
                            continue
                        except (concurrent.futures.BrokenExecutor, Exception) as e:
                            st.error(f"Chamber {chamber_idx} worker crashed: {e}")
                            completed += 1
                            progress_bar.progress(completed / total_to_process)
                            continue

                        completed += 1
                        progress_bar.progress(completed / total_to_process)

                        if success:
                            status_text.text(f"Finished Chamber {i} ({completed}/{total_to_process})")
                            all_summaries.extend(summary)

                            # Save progress incrementally
                            completed_chambers.add(i)
                            try:
                                progress_file.write_text(json.dumps(sorted(completed_chambers)))
                            except Exception:
                                pass

                            if all_summaries:
                                df = pd.DataFrame(all_summaries)
                                results_container.dataframe(df, use_container_width=True)
                        else:
                            st.error(f"Error in Chamber {i}: {result_path}")
                
            status_text.text("Analysis Complete!")
            st.session_state._analysis_output_dir = str(output_dir.resolve())
            st.balloons()
            
            # --- Download Section ---
            st.subheader("4. Downloads")
            results = sorted(list(output_dir.glob("*.csv")))
            
            # Zip all results
            zip_path = shutil.make_archive("courtship_results", "zip", output_dir)
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Download All Results (ZIP)",
                    data=f,
                    file_name="courtship_results.zip",
                    mime="application/zip",
                    key="dl_zip"
                )

            st.write("Individual Files:")
            for res in results:
                with open(res, "rb") as f:
                    st.download_button(
                        label=f"Download {res.name}",
                        data=f,
                        file_name=res.name,
                        mime="text/csv",
                        key=f"dl_{res.stem}"
                    )

def main():
    st.title("Fruit Fly Courtship Analysis System")
    # st.caption("for my lil baby <3")
    st.markdown("""
    **Python-based End-to-End Pipeline**
    1. Upload Video
    2. Detect Chambers & Crop
    3. Track & Classify Behaviors
    4. Download Results
    """)

    tab_analysis, tab_verify = st.tabs(["Analysis", "Verification"])

    with tab_analysis:
        _run_analysis()

    with tab_verify:
        verification_viewer()


if __name__ == "__main__":
    main()
