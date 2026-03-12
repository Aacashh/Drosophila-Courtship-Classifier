# Drosophila Courtship Behavior Analysis System

A Python-based video analysis pipeline for automated detection and classification of *Drosophila melanogaster* (fruit fly) courtship behaviors. The system provides a Streamlit web interface for processing multi-chamber arena videos, tracking individual flies, and classifying five distinct courtship behaviors.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Launching the Application](#launching-the-application)
  - [Step 1: Load a Video and Detect Chambers](#step-1-load-a-video-and-detect-chambers)
  - [Step 2: Configure Analysis Settings](#step-2-configure-analysis-settings)
  - [Step 3: Run the Analysis](#step-3-run-the-analysis)
  - [Step 4: Download Results](#step-4-download-results)
  - [Verification Tab](#verification-tab)
- [Output Files](#output-files)
- [Classification Details](#classification-details)
  - [Heuristic (Rule-Based) Classifier](#heuristic-rule-based-classifier)
  - [JAABA Classifier Support](#jaaba-classifier-support)
- [Project Architecture](#project-architecture)
- [Tuning and Improving Results](#tuning-and-improving-results)
  - [Adjusting Heuristic Thresholds](#adjusting-heuristic-thresholds)
  - [Improving Tracking Quality](#improving-tracking-quality)
  - [Using JAABA Classifiers](#using-jaaba-classifiers)
  - [Contributing New Behaviors](#contributing-new-behaviors)
- [Troubleshooting](#troubleshooting)

---

## Overview

This tool is designed for researchers studying *Drosophila* mating behavior. It processes videos of multi-chamber courtship arenas (typically 4x3 grids of 12 chambers, each containing one male and one female fly) and produces frame-level behavior annotations with summary statistics.

**Pipeline stages:**

1. **Chamber Detection** -- Automatically locates individual chambers in the arena video frame.
2. **Fly Tracking** -- Segments and tracks both flies per chamber using background subtraction and ellipse fitting.
3. **Feature Extraction** -- Computes motion, morphological, and social features (velocity, wing angle, inter-fly distance, facing angle, etc.).
4. **Behavior Classification** -- Classifies five courtship behaviors using configurable heuristic rules or pre-trained JAABA boosting classifiers.
5. **Export** -- Generates per-chamber CSV files with bout annotations and summary statistics (Courtship Index, latencies, bout counts).

---

## Features

- **Web-based UI** via Streamlit -- no command-line expertise required
- **Automatic chamber detection** with manual override options (grid builder, raw ROI editor)
- **Robust fly tracking** with identity maintenance across frames (Hungarian matching + temporal hysteresis)
- **Five courtship behaviors**: Wing Extension, Following, Circling, Attempted Copulation, Copulation
- **Sex inference** from behavioral data (wing extension frequency)
- **Video stabilization** (optional ECC-based alignment)
- **Parallel processing** -- analyze multiple chambers simultaneously (up to 4 workers)
- **Verification tab** -- review results with timeline visualizations and bout GIF previews
- **JAABA compatibility** -- load and run pre-trained JAABA boosting classifiers

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.10 or 3.11 |
| RAM | 4 GB | 8+ GB |
| Disk Space | 500 MB (+ video files) | 2+ GB |
| OS | Windows 10, macOS 10.15, Linux | Any modern OS |
| FFmpeg | Required (system install) | Latest stable |

---

## Installation

### 1. Install Python

Download and install Python 3.10 or 3.11 from [python.org](https://www.python.org/downloads/).

During installation on Windows, **check the box** that says **"Add Python to PATH"**.

Verify your installation:

```bash
python --version
```

### 2. Install FFmpeg

FFmpeg is required for video processing (cropping, stabilization).

**Windows:**
1. Download from [ffmpeg.org/download.html](https://ffmpeg.org/download.html) (choose "Windows builds" from gyan.dev or BtbN).
2. Extract the archive and add the `bin/` folder to your system PATH.
3. Verify: `ffmpeg -version`

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

### 3. Clone or Download the Repository

```bash
git clone <repository-url>
cd Courtship_Analysis_Python
```

Or download and extract the ZIP file from the repository.

### 4. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `streamlit>=1.30.0` | Web application framework |
| `opencv-python-headless` | Computer vision (tracking, segmentation) |
| `numpy>=1.24.0` | Numerical computing |
| `scipy>=1.10.0` | Scientific computing (filters, optimization) |
| `pandas>=2.0.0` | Data manipulation and CSV export |
| `ffmpeg-python>=0.2.0` | FFmpeg Python bindings for video cropping |
| `h5py>=3.10.0` | HDF5 file support (JAABA classifiers) |
| `matplotlib>=3.7.0` | Plotting and visualization |
| `Pillow>=9.0.0` | Image processing |

### 6. Verify Installation

```bash
streamlit hello
```

If a browser window opens with the Streamlit demo, your installation is complete.

---

## Usage Guide

### Launching the Application

```bash
streamlit run app.py
```

Your default web browser will open to `http://localhost:8501`. The interface has two main tabs: **Analysis** and **Verification**.

### Step 1: Load a Video and Detect Chambers

1. In the **Analysis** tab, you will see **Step 1: Detect Chambers**.
2. Provide your video by either:
   - Uploading a file through the file uploader (supports MP4, AVI, MOV, MKV; up to 4 GB), or
   - Entering a local file path directly.
3. Click **"Detect Chambers"** to automatically locate individual chambers in the first video frame.
4. The detected chambers are displayed as numbered rectangles overlaid on the frame.

**If auto-detection fails or is inaccurate**, you have three manual options:

- **Re-run Auto Detection** -- Click to retry with the same algorithm.
- **Manual Grid Builder** -- Specify the number of rows, columns, chamber dimensions (width x height), and gap sizes. The tool generates a uniform grid.
- **Raw ROI Editor** -- Enter chamber coordinates directly in `x, y, width, height` format (one chamber per line).

### Step 2: Configure Analysis Settings

Use the **sidebar** on the left to configure:

**Analysis Method:**
- **Heuristic (Built-in Rules)** -- Default. Uses configurable threshold-based rules. No external files needed.
- **JAABA Classifiers** -- Requires pre-trained `.jab` files. Can load from a folder or upload individual files.

**Calibration:**
- **Pixels per mm** -- Spatial calibration factor (default: 15.0). Measure this from your arena image (e.g., count pixels across a known distance). Accurate calibration is critical for velocity and distance thresholds.

**Heuristic Thresholds (expandable section):**
- All behavior thresholds can be tuned from the sidebar. See [Tuning and Improving Results](#tuning-and-improving-results) for details.

**Video Stabilization:**
- Enable this checkbox if your video has camera shake or drift. Uses ECC (Enhanced Correlation Coefficient) alignment. Adds processing time but improves tracking accuracy.

**Chamber Selection:**
- Select which chambers to analyze using checkboxes. Use "Select All" / "Deselect All" for convenience.

### Step 3: Run the Analysis

1. Click the **"Run Analysis"** button.
2. A progress bar shows overall completion.
3. Chambers are processed in parallel (up to 4 simultaneous workers).
4. Results appear in a live-updating table showing:
   - Chamber number
   - Fly label (with inferred sex: Male/Female)
   - Courtship Index (%)
   - Per-behavior bout counts and total durations

### Step 4: Download Results

After analysis completes:

- **Download All Results (ZIP)** -- A single ZIP file containing all chamber CSVs.
- **Individual Chamber CSVs** -- Download results for specific chambers.

Results are also saved locally in the `analysis_results/` or `courtship_results/` directory.

### Verification Tab

The **Verification** tab lets you review and validate results:

1. **Select a results directory** -- The app auto-detects `courtship_results/` and `analysis_results/`.
2. **Choose a chamber** to inspect.
3. **View the behavior timeline** -- A stacked bar chart showing when each behavior occurred over the video duration.
4. **Browse individual bouts** -- Select a behavior and bout to see:
   - Start/end times and duration
   - A GIF preview of the video during that bout
5. **Summary statistics table** -- Courtship Index, latency metrics, per-behavior counts.
6. **Full results table** -- Every detected bout with timestamps.
7. **Filter by male only** -- Focus on the male fly's behaviors.

---

## Output Files

For each analyzed chamber, two CSV files are generated:

### `chamber_N_results.csv` -- Detailed Bout List

| Column | Description |
|--------|-------------|
| Behavior | Behavior name (WingExt, Following, Circling, Attempted_Copulation, Copulation) |
| Fly | Fly label with inferred sex (e.g., "Fly 0 (Male)") |
| Start_Time (s) | Bout start time in seconds |
| End_Time (s) | Bout end time in seconds |
| Duration (s) | Bout duration in seconds |
| Start_Frame | Bout start frame number |
| End_Frame | Bout end frame number |

### `chamber_N_summary.csv` -- Summary Statistics

| Metric | Description |
|--------|-------------|
| Courtship_Index (%) | Percentage of frames with any courtship behavior |
| Latency_First_Courtship (s) | Time to first courtship event |
| Latency_Copulation (s) | Time to first copulation event |
| {Behavior}_Count | Number of bouts for each behavior |
| {Behavior}_Duration (s) | Total duration for each behavior |
| Video_Duration (s) | Total video length |

---

## Classification Details

### Heuristic (Rule-Based) Classifier

Five behaviors are classified using the following default criteria:

#### Wing Extension
- Wing angle > **60 degrees**
- Minimum bout duration: **1.0 second**
- Indicates courtship song production

#### Following
- Forward velocity (v_par) > **2.0 mm/s**
- Facing angle toward partner < **45 degrees**
- Inter-fly distance: **2.0 -- 8.0 mm**
- Minimum bout duration: **1.0 second**

#### Circling
- Lateral velocity (v_perp) > **2.0 mm/s**
- Inter-fly distance < **10.0 mm**
- Facing angle: **45 -- 135 degrees** (sideways orientation)
- Minimum bout duration: **1.0 second**

#### Attempted Copulation
- Nose-to-partner distance < **0.5 mm**
- Forward velocity > **1.0 mm/s**
- Minimum bout duration: **0.5 seconds**

#### Copulation
- Inter-fly distance < **2.0 mm**
- Velocity < **0.3 mm/s**
- Positional stability < **0.5 mm** over window
- Minimum bout duration: **8.0 seconds**

**Post-processing:**
- Gaussian smoothing applied to raw scores to reduce noise
- Adjacent bouts within **0.5 seconds** are merged
- Mutual exclusion priority: Copulation > Attempted Copulation > Following > Circling (Wing Extension is exempt and can co-occur with other behaviors)

### JAABA Classifier Support

The system can load pre-trained JAABA (Janelia Automatic Animal Behavior Annotator) classifiers:

- Supports both MATLAB v7 (`.mat`) and HDF5 v7.3 (`.jab`) file formats
- Boosting classifiers with decision stump weak learners
- Compatible with JAABA's token-based feature system (windowed statistics, transforms)
- Load classifiers from a folder or upload individual `.jab` files

---

## Project Architecture

```
Courtship_Analysis_Python/
├── app.py                        # Streamlit web UI (entry point)
│
├── tracking/
│   ├── tracker.py                # MOG2 background subtraction
│   │                             #   Ellipse fitting for fly body
│   │                             #   Hungarian identity assignment
│   │                             #   NaN interpolation & smoothing
│   └── features.py              # Feature extraction
│                                 #   Social: distance, facing, nose/tail
│                                 #   Motion: v_par, v_perp, angular vel
│                                 #   Head/tail resolution with hysteresis
│
├── classification/
│   ├── heuristic.py             # Rule-based classifier (5 behaviors)
│   ├── inference.py             # JAABA boosting classifier inference
│   └── jab_parser.py            # JAABA .jab file parser
│
├── utils/
│   ├── video.py                 # Chamber detection, crop, stabilize
│   └── export.py                # CSV export, sex inference, summaries
│
├── reference/                    # Test data (not tracked in git)
│   ├── movie.mp4                # Sample 10-minute arena video
│   └── *.csv                    # Manual annotations for validation
│
├── requirements.txt             # Python dependencies
└── .streamlit/
    └── config.toml              # Streamlit config (4 GB upload limit)
```

---

## Tuning and Improving Results

### Adjusting Heuristic Thresholds

The most impactful improvement you can make is tuning the classification thresholds to match your specific experimental setup. All thresholds are adjustable from the sidebar in the Streamlit UI.

**Key parameters to tune:**

| Parameter | Effect of Increasing | Effect of Decreasing |
|-----------|---------------------|---------------------|
| Wing angle threshold | Fewer, more confident wing extension detections | More sensitive, may include noise |
| Min bout duration | Fewer, longer bouts; removes brief false positives | Captures shorter behavioral events |
| Distance thresholds | More permissive spatial criteria | Stricter proximity requirements |
| Velocity thresholds | Requires faster movement to qualify | Captures slower behaviors |
| Merge gap | More aggressive bout merging | Keeps bouts separate |
| Pixels per mm | All spatial thresholds effectively shrink | All spatial thresholds effectively grow |

**Calibration tip:** The **pixels per mm** setting is the single most important parameter. If your velocity and distance thresholds seem too sensitive or too strict, recalibrate this value first. Measure a known distance in your arena (e.g., chamber width) and divide the pixel measurement by the real-world distance in mm.

### Improving Tracking Quality

1. **Use video stabilization** for videos with any camera movement. This is enabled via a checkbox in the analysis settings.
2. **Ensure good contrast** between flies and the arena background. The tracker uses background subtraction (MOG2), which works best with consistent, uniform backgrounds.
3. **Avoid reflections and shadows** in the arena -- these create false contours that confuse the tracker.
4. **Consistent lighting** across the arena improves segmentation quality. Uneven lighting causes some chambers to track better than others.
5. **Frame rate matters** -- The tracker interpolates short gaps (up to 5 frames). Higher frame rates give more temporal resolution but increase processing time.

### Using JAABA Classifiers

If you have JAABA classifiers trained on your specific fly strains and experimental conditions, they will likely outperform the heuristic rules:

1. Train classifiers in JAABA (MATLAB) on manually annotated data from your setup.
2. Export the `.jab` files.
3. In the Streamlit sidebar, switch the analysis method to **"JAABA Classifiers"**.
4. Load your `.jab` files from a folder or upload them directly.
5. The system extracts feature tokens from the `.jab` files and computes matching features for inference.

**Token files:** If the feature names are not embedded in your `.jab` files, place a `tokens.json` file alongside them mapping feature indices to token definitions.

### Contributing New Behaviors

To add a new behavior to the heuristic classifier:

1. **Define the behavior rules** in `classification/heuristic.py`:
   - Add threshold parameters to `DEFAULT_PARAMS`
   - Add a scoring function in `classify_heuristic()` that produces a binary array
   - Add the behavior to the mutual exclusion priority list if needed

2. **Add required features** in `tracking/features.py`:
   - If the behavior needs new features not already computed, add them to `build_heuristic_features()`

3. **Update the UI** in `app.py`:
   - Add threshold controls to the sidebar settings section
   - Add the behavior to the results display and verification tab

4. **Update the export** in `utils/export.py`:
   - The export functions are behavior-agnostic (they iterate over whatever behaviors are in the results dict), so new behaviors should work automatically

---

## Troubleshooting

### "No chambers detected"

- The auto-detection algorithm expects white/light chambers on a darker background. If your arena has different contrast, use the **Manual Grid Builder** or **Raw ROI Editor**.
- Try adjusting your video's brightness/contrast before loading.
- Ensure the first frame of your video shows the full arena clearly (no occlusion, no motion blur).

### Tracking is noisy or flies swap identities

- Enable **video stabilization** if not already enabled.
- Check that your **pixels per mm** calibration is accurate.
- The tracker uses a max-distance threshold (25% of chamber dimension) for identity assignment. If flies move very quickly between frames, they may exceed this threshold and swap. Consider using a higher frame rate video.
- Overlap frames (where both flies are touching) are automatically flagged and excluded from classification.

### Very few or no behaviors detected

- Your **pixels per mm** calibration may be wrong, causing velocity and distance thresholds to be ineffective. Recalibrate first.
- Lower the minimum bout durations to capture shorter events.
- Lower the velocity thresholds if your flies are moving slowly.
- Check the tracking output visually using the Verification tab to ensure flies are being tracked correctly.

### "FFmpeg not found" or cropping errors

- Ensure FFmpeg is installed and available on your system PATH.
- Test by running `ffmpeg -version` in your terminal.
- On Windows, you may need to restart your terminal/IDE after adding FFmpeg to PATH.

### Application crashes or runs out of memory

- Large videos (> 1 hour, high resolution) consume significant RAM. Close other applications.
- Process fewer chambers at a time by deselecting some in the chamber selection UI.
- Reduce the number of parallel workers (currently hardcoded to 4 in `app.py`; can be modified in the `ProcessPoolExecutor` call).

### Streamlit upload fails for large files

- The maximum upload size is configured to **4 GB** in `.streamlit/config.toml`. For larger files, use the **local file path** option instead of uploading.

---

## License

This project is developed for research purposes. Contact the repository maintainers for licensing information.
