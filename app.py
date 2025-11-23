import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time
import concurrent.futures
import pandas as pd

from utils.video import detect_chambers, crop_video, stabilize_video
from tracking.tracker import track_two_flies
from tracking.features import build_feature_matrix
from classification.jab_parser import load_jab_models
from classification.inference import eval_boosting, scores_to_bouts
from classification.heuristic import HeuristicClassifier
from utils.export import export_to_csv, infer_sex

st.set_page_config(page_title="Courtship Analysis", layout="wide")

def process_single_chamber(i, roi, video_path, output_dir, do_stabilize, analysis_mode, models, px_per_mm):
    """Worker function for parallel processing."""
    try:
        # 1. Crop
        chamber_video = output_dir / f"chamber_{i}.mp4"
        crop_video(Path(video_path), chamber_video, roi)
        
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
            bouts_by_behavior = HeuristicClassifier.classify(tracks, fps, px_per_mm)
        
        # 5. Export
        fly_stats = infer_sex(bouts_by_behavior)
        csv_path = output_dir / f"chamber_{i}_results.csv"
        export_to_csv(csv_path, bouts_by_behavior, fps, fly_stats)
        
        # Return summary for visualization
        summary = []
        for beh, bouts_list in bouts_by_behavior.items():
            for fly_idx, bouts in enumerate(bouts_list):
                duration = sum((e - s + 1) for s, e in bouts) / fps
                summary.append({
                    'Chamber': i,
                    'Fly': fly_idx,
                    'Behavior': beh,
                    'Count': len(bouts),
                    'Total Duration (s)': round(duration, 2)
                })
        
        return i, True, csv_path, summary
    except Exception as e:
        return i, False, str(e), []

def main():
    st.title("🦟 Fruit Fly Courtship Analysis System")
    st.markdown("""
    **Python-based End-to-End Pipeline**
    1. Upload Video
    2. Detect Chambers & Crop
    3. Track & Classify Behaviors
    4. Download Results
    """)

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
    
    # --- Main Workflow ---
    
    uploaded_video = st.file_uploader("Choose a Video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is not None:
        # Save uploaded video to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        tfile.close() # Ensure data is flushed to disk
        video_path = tfile.name
        
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
                st.session_state.rois = detect_chambers(first_frame)
                
            # --- Interactive Grid Adjustment ---
            st.write("### Fine-tune Detection")
            if st.button("Re-run Auto Detection"):
                st.session_state.rois = detect_chambers(first_frame)
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
            
            # Create output dir
            output_dir = Path("analysis_results")
            output_dir.mkdir(exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing...")
            
            # Visualization containers
            st.subheader("3. Real-time Results")
            results_container = st.container()
            
            # Parallel Processing
            max_workers = min(4, len(selected_indices)) 
            
            all_summaries = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i in selected_indices:
                    roi = st.session_state.rois[i]
                    futures.append(
                        executor.submit(process_single_chamber, i, roi, video_path, output_dir, do_stabilize, analysis_mode, models, px_per_mm)
                    )
                
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    i, success, result_path, summary = future.result()
                    completed += 1
                    progress_bar.progress(completed / len(selected_indices))
                    
                    if success:
                        status_text.text(f"Finished Chamber {i}")
                        all_summaries.extend(summary)
                        
                        # Update Table
                        if all_summaries:
                            df = pd.DataFrame(all_summaries)
                            # Pivot for cleaner view? Or just raw table
                            results_container.dataframe(df, use_container_width=True)
                            
                    else:
                        st.error(f"Error in Chamber {i}: {result_path}")
                
            status_text.text("Analysis Complete!")
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
                    mime="application/zip"
                )
            
            st.write("Individual Files:")
            for res in results:
                with open(res, "rb") as f:
                    st.download_button(
                        label=f"Download {res.name}",
                        data=f,
                        file_name=res.name,
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()
