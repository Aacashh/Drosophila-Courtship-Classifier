"""
Generate comprehensive technical documentation PDF for the
Drosophila Courtship Behavior Analysis Pipeline.
"""

from fpdf import FPDF
import os
import textwrap

class DocPDF(FPDF):
    """Custom PDF class with headers, footers, and helper methods."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Drosophila Courtship Behavior Analysis Pipeline -- Technical Documentation", align="C")
            self.ln(4)
            self.set_draw_color(180, 180, 180)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, number, title):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 60, 120)
        self.ln(6)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(6)
        self.set_text_color(0, 0, 0)

    def section_title(self, number, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(40, 80, 140)
        self.ln(4)
        self.cell(0, 8, f"{number} {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text.strip())
        self.ln(2)

    def bullet_point(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text.strip())
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(30, 30, 30)
        lines = text.strip().split("\n")
        for line in lines:
            self.cell(0, 5, "  " + line, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)
        self.set_font("Helvetica", "", 10)

    def table_row(self, cells, widths, bold=False, fill=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        if fill:
            self.set_fill_color(220, 230, 245)
        h = 6
        for i, (cell, w) in enumerate(zip(cells, widths)):
            self.cell(w, h, str(cell), border=1, fill=fill)
        self.ln(h)

    def italic_text(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text.strip())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def bold_inline(self, label, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        lw = self.get_string_width(label + " ") + 2
        self.cell(lw, 5.5, label)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text.strip())
        self.ln(1)


def build_pdf():
    pdf = DocPDF()
    pdf.alias_nb_pages()

    # ===== TITLE PAGE =====
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 14, "Drosophila Courtship Behavior", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 14, "Analysis Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Technical Documentation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(20, 60, 120)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "A Python-Based Computer Vision System for Automated", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Detection and Classification of Fruit Fly Courtship Behaviors", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "in Multi-Chamber Arena Experiments", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 7, "Version 1.0", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    # ===== TABLE OF CONTENTS =====
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    toc = [
        ("1", "Motivation and Background", 3),
        ("  1.1", "The Problem", 3),
        ("  1.2", "Existing Approaches and Their Limitations", 3),
        ("  1.3", "Design Goals", 3),
        ("2", "System Architecture", 4),
        ("  2.1", "Pipeline Overview", 4),
        ("  2.2", "Module Organization", 4),
        ("3", "Chamber Detection and Video Preprocessing", 5),
        ("  3.1", "Multi-Panel Grid Detection", 5),
        ("  3.2", "Grid Snapping and Regularization", 5),
        ("  3.3", "Video Stabilization via ECC", 5),
        ("4", "Fly Tracking", 6),
        ("  4.1", "Background Subtraction (MOG2)", 6),
        ("  4.2", "Ellipse Fitting and Body Modeling", 6),
        ("  4.3", "Head-Tail Orientation Resolution", 6),
        ("  4.4", "Identity Assignment via the Hungarian Algorithm", 7),
        ("  4.5", "Post-Processing: Gap Interpolation and Smoothing", 7),
        ("5", "Feature Extraction", 8),
        ("  5.1", "Kinematic Features", 8),
        ("  5.2", "Social Geometry Features", 8),
        ("  5.3", "JAABA-Compatible Window Features", 8),
        ("6", "Behavior Classification", 9),
        ("  6.1", "Heuristic (Rule-Based) Classifier", 9),
        ("  6.2", "JAABA Boosting Classifier", 10),
        ("  6.3", "Temporal Smoothing and Bout Detection", 10),
        ("  6.4", "Mutual Exclusion and Priority Resolution", 10),
        ("7", "Output and Summary Statistics", 11),
        ("  7.1", "Courtship Index", 11),
        ("  7.2", "Sex Inference", 11),
        ("  7.3", "CSV Export Format", 11),
        ("8", "Installation and Usage", 12),
        ("9", "References", 13),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    for num, title, _page in toc:
        style = "B" if not num.startswith(" ") else ""
        pdf.set_font("Helvetica", style, 11)
        pdf.cell(0, 6.5, f"{num}   {title}", new_x="LMARGIN", new_y="NEXT")

    # ===== SECTION 1: MOTIVATION =====
    pdf.add_page()
    pdf.chapter_title("1", "Motivation and Background")

    pdf.section_title("1.1", "The Problem")
    pdf.body_text(
        "Drosophila melanogaster courtship behavior constitutes one of the most extensively "
        "studied innate behavioral repertoires in genetics and neuroscience. Male flies perform "
        "a stereotyped sequence of actions directed toward females: orientation toward the target, "
        "following, unilateral wing extension to produce a species-specific courtship song, "
        "circling, licking, attempted copulation, and copulation (Hall, 1994; Greenspan and "
        "Ferveur, 2000). Each of these behavioral elements is under genetic control and can be "
        "selectively disrupted by mutations in genes such as fruitless (fru) and doublesex (dsx) "
        "(Yamamoto and Koganezawa, 2013)."
    )
    pdf.body_text(
        "Quantification of courtship behavior is essential for phenotyping mutant lines, assessing "
        "the effects of neural circuit manipulations, and studying the sensory ecology of mate "
        "choice. The standard metric, the Courtship Index (CI), was introduced by Siegel and "
        "Hall (1979) and is defined as the fraction of observation time during which the male "
        "engages in any courtship-related activity. Historically, scoring courtship has been "
        "performed manually by trained observers viewing recorded videos, a process that is "
        "labor-intensive, subject to inter-observer variability, and poorly scalable to high-"
        "throughput experimental designs."
    )

    pdf.section_title("1.2", "Existing Approaches and Their Limitations")
    pdf.body_text(
        "Several automated systems have been developed to address this bottleneck. Ctrax "
        "(Branson et al., 2009) provides robust multi-fly tracking using background subtraction "
        "and ellipse fitting but does not classify behaviors. FlyTracker (Eyjolfsdottir et al., "
        "2014) extends tracking to include social feature computation for pairs of interacting "
        "flies. JAABA (Kabra et al., 2013) builds upon these trackers by providing interactive "
        "machine-learning classifiers trained via GentleBoost (Friedman et al., 2000) on "
        "windowed per-frame features."
    )
    pdf.body_text(
        "While powerful, these tools have practical limitations. They typically require MATLAB "
        "licenses, involve multi-step workflows across separate applications, and demand "
        "researcher-labeled training data for each new experimental setup. For laboratories "
        "running routine courtship assays with standardized arenas, a self-contained system "
        "that integrates tracking, feature extraction, and rule-based classification into a "
        "single pipeline would be of considerable practical value."
    )

    pdf.section_title("1.3", "Design Goals")
    pdf.body_text(
        "This system was designed with the following objectives:"
    )
    pdf.bullet_point(
        "Provide an end-to-end pipeline that accepts a raw multi-chamber arena video and "
        "produces per-chamber courtship annotations and summary statistics with no manual intervention."
    )
    pdf.bullet_point(
        "Support both heuristic (rule-based) classification using literature-derived thresholds "
        "and JAABA-trained boosting classifiers for laboratories that have existing .jab model files."
    )
    pdf.bullet_point(
        "Run entirely in Python with open-source dependencies, requiring no MATLAB license."
    )
    pdf.bullet_point(
        "Expose a web-based interface (Streamlit) that allows parameter tuning, chamber "
        "selection, and visual verification of results."
    )
    pdf.bullet_point(
        "Produce output compatible with downstream statistical analysis, including per-bout "
        "CSVs and summary statistics with Courtship Index, latency metrics, and sex inference."
    )

    # ===== SECTION 2: ARCHITECTURE =====
    pdf.add_page()
    pdf.chapter_title("2", "System Architecture")

    pdf.section_title("2.1", "Pipeline Overview")
    pdf.body_text(
        "The analysis pipeline processes a single multi-chamber arena video through five "
        "sequential stages:"
    )
    pdf.body_text(
        "Stage 1: Chamber Detection. The first frame (or a median of several frames for "
        "robustness to occlusions) is analyzed to locate individual chambers within the arena "
        "grid. Detected regions are snapped to a regular grid to ensure uniform, edge-inclusive "
        "bounding boxes."
    )
    pdf.body_text(
        "Stage 2: Video Preprocessing. Each chamber is cropped from the source video. An "
        "optional stabilization step uses Enhanced Correlation Coefficient (ECC) alignment "
        "(Evangelidis and Psarakis, 2008) to correct for camera vibration or arena drift."
    )
    pdf.body_text(
        "Stage 3: Fly Tracking. Within each cropped chamber video, two flies are tracked "
        "frame-by-frame using MOG2 background subtraction (Zivkovic, 2004; Zivkovic and "
        "van der Heijden, 2006), contour detection, and ellipse fitting (Fitzgibbon et al., "
        "1999). Frame-to-frame identity is maintained using the Hungarian algorithm "
        "(Kuhn, 1955; Munkres, 1957)."
    )
    pdf.body_text(
        "Stage 4: Feature Extraction and Behavior Classification. Kinematic and social "
        "geometry features are computed from the tracking data. Behaviors are classified "
        "either through rule-based heuristics with literature-derived thresholds or through "
        "JAABA boosting classifiers."
    )
    pdf.body_text(
        "Stage 5: Export. Per-chamber CSV files are generated containing bout-level "
        "annotations (behavior, fly identity, start/end times, duration) along with summary "
        "statistics (Courtship Index, latency to first courtship, latency to copulation)."
    )

    pdf.section_title("2.2", "Module Organization")
    pdf.body_text("The codebase is organized into the following modules:")
    w1, w2 = 55, 120
    pdf.table_row(["Module", "Description"], [w1, w2], bold=True, fill=True)
    pdf.table_row(["app.py", "Streamlit UI, orchestration, parallel processing"], [w1, w2])
    pdf.table_row(["tracking/tracker.py", "MOG2 segmentation, ellipse fitting, identity assignment"], [w1, w2])
    pdf.table_row(["tracking/features.py", "Kinematic/social features, JAABA window features"], [w1, w2])
    pdf.table_row(["classification/heuristic.py", "Rule-based behavior classifier (5 behaviors)"], [w1, w2])
    pdf.table_row(["classification/inference.py", "Boosting classifier evaluation, bout detection"], [w1, w2])
    pdf.table_row(["classification/jab_parser.py", "JAABA .jab file parser (v7 and v7.3 HDF5)"], [w1, w2])
    pdf.table_row(["utils/video.py", "Chamber detection, cropping, ECC stabilization"], [w1, w2])
    pdf.table_row(["utils/export.py", "CSV export, sex inference, summary statistics"], [w1, w2])

    # ===== SECTION 3: CHAMBER DETECTION =====
    pdf.add_page()
    pdf.chapter_title("3", "Chamber Detection and Video Preprocessing")

    pdf.section_title("3.1", "Multi-Panel Grid Detection")
    pdf.body_text(
        "Multi-chamber courtship arenas typically consist of one or two grid panels, each "
        "containing a regular array of small rectangular chambers. The arena may be imaged "
        "with varying numbers of chambers and spatial layouts. Chamber detection proceeds "
        "hierarchically: first, large grid regions are identified, then individual chambers "
        "are detected within each region."
    )
    pdf.body_text(
        "To mitigate transient occlusions (e.g., a researcher's hand entering the frame "
        "during video setup), the system samples multiple frames from the first quarter "
        "of the video and computes the pixel-wise median. This temporal median provides "
        "a stable reference frame in which brief occlusions are suppressed."
    )
    pdf.body_text(
        "Grid regions are detected by applying Otsu thresholding to the grayscale reference "
        "frame, followed by heavy morphological closing (kernel size proportional to image "
        "dimensions) to merge all chambers within a panel into a single connected component. "
        "A subsequent morphological opening removes thin noise bridges between physically "
        "separate grids. Contours of the resulting blobs are filtered by area (at least 5% of "
        "the total frame area) and aspect ratio (between 0.3 and 3.0) to yield grid-level "
        "bounding boxes."
    )

    pdf.section_title("3.2", "Grid Snapping and Regularization")
    pdf.body_text(
        "Within each grid region, individual chambers are detected by a second round of "
        "Otsu thresholding and contour analysis at finer scale, with area-based filtering "
        "(0.5% to 25% of the grid region) and aspect-ratio constraints. The resulting "
        "candidate bounding boxes are validated for size consistency: detections whose area "
        "deviates by more than a factor of 2.5 from the median are rejected as outliers "
        "(debris, hands, shadows)."
    )
    pdf.body_text(
        "The remaining candidates are then snapped to a regular grid. The algorithm clusters "
        "candidate centers along the X and Y axes using the median box dimensions as a gap "
        "threshold. Row and column centers are computed from these clusters, and the cell "
        "size is set to the 75th percentile of detected dimensions (slightly larger than the "
        "median, to avoid clipping chamber edges that contour detection may have missed). A "
        "uniform grid of bounding boxes is reconstructed from these row/column centers and "
        "the standardized cell size, with configurable padding (default 15 pixels) to ensure "
        "complete coverage."
    )

    pdf.section_title("3.3", "Video Stabilization via ECC")
    pdf.body_text(
        "After cropping, each chamber video may optionally be stabilized to correct for "
        "camera vibration or slow arena drift. The stabilization module aligns every frame "
        "to the first frame using the Enhanced Correlation Coefficient (ECC) method of "
        "Evangelidis and Psarakis (2008), implemented in OpenCV as cv2.findTransformECC(). "
        "The ECC criterion is invariant to global photometric changes (brightness, contrast), "
        "making it suitable for experimental recordings with variable illumination."
    )
    pdf.body_text(
        "Two optimizations are applied for computational efficiency. First, ECC alignment is "
        "computed on half-resolution images, reducing per-frame computation by approximately "
        "4x; the resulting translation parameters are then scaled back to full resolution. "
        "Second, when the estimated motion magnitude falls below 0.5 pixels, the current "
        "warp matrix is reused for the next 5 frames without recomputing ECC. A Euclidean "
        "motion model (rotation plus translation) is used, with a maximum of 20 iterations "
        "and a convergence threshold of 10^-3."
    )

    # ===== SECTION 4: TRACKING =====
    pdf.add_page()
    pdf.chapter_title("4", "Fly Tracking")

    pdf.section_title("4.1", "Background Subtraction (MOG2)")
    pdf.body_text(
        "Foreground segmentation is performed using the Gaussian Mixture Model (MOG2) "
        "algorithm of Zivkovic (2004) and Zivkovic and van der Heijden (2006). Each pixel "
        "is modeled as a mixture of Gaussian distributions whose parameters are updated "
        "recursively as new frames are processed. The algorithm automatically adapts the "
        "number of mixture components per pixel, typically using between 3 and 5 components "
        "depending on the local complexity of the background. This adaptive behavior is "
        "well suited to courtship arenas where the background is largely static but may "
        "exhibit slight variations due to lighting changes or arena texture."
    )
    pdf.body_text(
        "The MOG2 background subtractor is configured with a history window of 100 frames, "
        "a variance threshold of 20, and shadow detection disabled (as the white-background "
        "arenas used in courtship assays do not produce strong shadows). The resulting "
        "foreground mask is refined through morphological opening (3x3 elliptical kernel) "
        "to remove salt-and-pepper noise, followed by morphological closing (5x5 elliptical "
        "kernel) to fill small holes within fly body regions. A binary threshold at value 127 "
        "is applied to produce the final segmentation mask."
    )

    pdf.section_title("4.2", "Ellipse Fitting and Body Modeling")
    pdf.body_text(
        "External contours are extracted from the binary segmentation mask and filtered by "
        "area (minimum 0.05% of the chamber dimensions) to remove noise. The two largest "
        "valid contours (corresponding to the two flies) are retained. For each contour "
        "with at least 5 points, an ellipse is fit using the direct least-squares method "
        "of Fitzgibbon et al. (1999), as implemented in OpenCV's cv2.fitEllipse(). This "
        "yields the centroid (x, y), semi-major axis (a), semi-minor axis (b), and "
        "orientation angle for each fly."
    )
    pdf.body_text(
        "The geometric wing angle is computed by subtracting the fitted body ellipse from "
        "the original fly contour, yielding a residual 'wings mask.' Connected components "
        "of this mask are filtered by size (between 5% and 50% of the body area) and "
        "proximity to the body centroid (within 2x the major axis length). The angle "
        "between the wing centroid and the body heading vector gives the wing extension "
        "angle."
    )

    pdf.section_title("4.3", "Head-Tail Orientation Resolution")
    pdf.body_text(
        "Ellipse fitting inherently carries a 180-degree ambiguity: the fitted orientation "
        "could point toward the head or the tail. This ambiguity is resolved through a "
        "combination of velocity-based and temporal-continuity cues."
    )
    pdf.body_text(
        "For each frame, two candidate orientations (theta and theta + pi) are evaluated. "
        "When the fly is moving above a speed threshold (set adaptively at 20% of the "
        "median speed across the video), the candidate whose direction is closer to the "
        "instantaneous velocity vector is preferred. When velocity cues conflict with "
        "temporal continuity (the orientation from the previous frame), a hysteresis "
        "mechanism resolves the conflict: velocity is preferred only when the angular "
        "difference between the two candidates exceeds 0.3 radians; otherwise, the "
        "previous orientation is propagated. When the fly is stationary, temporal continuity "
        "alone determines the orientation."
    )
    pdf.body_text(
        "This pixel-mass check is complemented at the detection level: for each frame, the "
        "foreground pixels within a region of interest around the fly centroid are projected "
        "along the ellipse's major axis. The half with greater cumulative intensity (after "
        "inverting the grayscale, so darker pixels carry more weight) is designated as the "
        "head direction. This two-stage approach (pixel mass at detection time, velocity-"
        "based hysteresis at the trajectory level) provides stable orientation estimates "
        "even during stationary periods or brief occlusions."
    )

    pdf.section_title("4.4", "Identity Assignment via the Hungarian Algorithm")
    pdf.body_text(
        "Frame-to-frame identity assignment ensures that the same physical fly retains the "
        "same label throughout the video. This is formulated as a bipartite assignment "
        "problem: given the positions of n tracked flies in the previous frame and m "
        "detections in the current frame, find the assignment that minimizes total "
        "displacement."
    )
    pdf.body_text(
        "The cost matrix is populated with Euclidean distances between each previous "
        "position and each current detection. The optimal assignment is found using the "
        "Hungarian algorithm (Kuhn, 1955; Munkres, 1957), via scipy.optimize.linear_sum_assignment(). "
        "A maximum distance threshold (set at 25% of the smaller chamber dimension) is "
        "enforced to prevent identity swaps when flies pass close to each other: matches "
        "exceeding this threshold are rejected, and the corresponding fly is marked as "
        "temporarily lost (NaN). When a fly is lost, its last known position is retained "
        "for matching in subsequent frames, enabling re-identification after brief occlusions."
    )

    pdf.section_title("4.5", "Post-Processing: Gap Interpolation and Smoothing")
    pdf.body_text(
        "After tracking, short NaN gaps (up to 5 frames) are interpolated. Positional "
        "features (x, y, semi-axis lengths, area) are linearly interpolated between the "
        "flanking valid frames. Orientation (theta) is interpolated using circular "
        "interpolation to avoid wraparound artifacts. Wing angle is set to zero during "
        "interpolated frames, as no reliable measurement is available."
    )
    pdf.body_text(
        "A light Gaussian smoothing (sigma = 1.5 frames) is applied to the x and y "
        "trajectories to reduce detection jitter without blurring fast movements. NaN "
        "regions are handled by temporary linear interpolation before smoothing, with "
        "NaN values restored afterward."
    )
    pdf.body_text(
        "Overlap detection flags are computed by comparing the centroid-to-centroid "
        "distance to the sum of the semi-major axes of both flies. Frames in which this "
        "distance is smaller than the summed body lengths are flagged as overlap frames. "
        "Downstream classifiers use these flags to zero out behavioral scores during "
        "overlaps, preventing false positive detections when flies are in physical contact "
        "and individual features cannot be reliably measured."
    )

    # ===== SECTION 5: FEATURES =====
    pdf.add_page()
    pdf.chapter_title("5", "Feature Extraction")

    pdf.section_title("5.1", "Kinematic Features")
    pdf.body_text(
        "Several kinematic features are computed from the tracked trajectories. All velocity "
        "computations apply Gaussian smoothing (sigma = 0.05 * fps frames) to the position "
        "traces prior to numerical differentiation, reducing amplification of detection noise "
        "by the gradient operator."
    )

    pdf.subsection_title("Forward and Lateral Velocity (v_par, v_perp)")
    pdf.body_text(
        "The velocity vector (vx, vy) is decomposed into components parallel and "
        "perpendicular to the fly's body axis. The forward (parallel) velocity v_par is "
        "the dot product of the velocity vector with the heading unit vector (cos(theta), "
        "sin(theta)). The lateral (perpendicular) velocity v_perp is the dot product with "
        "the orthogonal vector (-sin(theta), cos(theta)). These components distinguish "
        "between forward locomotion (relevant for following) and lateral movement (relevant "
        "for circling)."
    )

    pdf.subsection_title("Angular Velocity")
    pdf.body_text(
        "Angular velocity is computed as the frame-to-frame difference in orientation, "
        "wrapped to the interval [-pi, pi] and multiplied by the frame rate to yield "
        "radians per second. Gaussian smoothing (sigma = 0.05 * fps) is applied after "
        "differentiation."
    )

    pdf.subsection_title("Total Speed")
    pdf.body_text(
        "Total speed is the Euclidean magnitude of the smoothed velocity vector: "
        "sqrt(vx^2 + vy^2). This feature is used in copulation detection, where "
        "both flies are expected to be nearly stationary."
    )

    pdf.section_title("5.2", "Social Geometry Features")
    pdf.body_text(
        "Social features describe the spatial relationship between the focal fly and its "
        "partner. These are computed in the module tracking/features.py, function "
        "compute_social_features()."
    )

    pdf.subsection_title("Centroid-to-Centroid Distance (c2c)")
    pdf.body_text(
        "The Euclidean distance between the centroids of the two flies. This is the primary "
        "proximity measure and enters the thresholds for following, copulation, and circling."
    )

    pdf.subsection_title("Facing Angle")
    pdf.body_text(
        "The angle between the focal fly's heading direction and the bearing toward the "
        "partner's centroid, computed as bearing - theta and wrapped to [-pi, pi]. A facing "
        "angle near zero indicates the fly is oriented toward its partner; a value near pi "
        "indicates it is facing away."
    )

    pdf.subsection_title("Nose-to-Ellipse Distance (n2e)")
    pdf.body_text(
        "The distance from the focal fly's nose (tip of the body ellipse along the heading "
        "direction) to the surface of the partner's body ellipse. The partner ellipse's "
        "radius in the direction of the nose point is computed analytically using the "
        "formula r(phi) = ab / sqrt((b*cos(phi))^2 + (a*sin(phi))^2), where phi is the "
        "angle relative to the partner's orientation. The n2e distance is the raw distance "
        "from nose to partner centroid minus this directional radius. This feature is "
        "central to attempted copulation detection, where the male's nose must be in "
        "contact with the female's body."
    )

    pdf.subsection_title("Wing Angle")
    pdf.body_text(
        "The geometric angle of the largest wing blob relative to the body axis, extracted "
        "during the tracking stage (Section 4.2). This feature drives wing extension detection."
    )

    pdf.section_title("5.3", "JAABA-Compatible Window Features")
    pdf.body_text(
        "When JAABA boosting classifiers are used, the system constructs a feature matrix "
        "compatible with JAABA's window feature specification. Each window feature is "
        "defined by a base per-frame time series (e.g., velocity, facing angle, wing angle), "
        "an optional transform (absolute value, flip, relative), a window statistic "
        "(mean, min, max, standard deviation, change, z-score of neighbors, difference "
        "from neighbor mean/min/max), a temporal radius, and a temporal offset."
    )
    pdf.body_text(
        "The feature name tokens are parsed from the JAABA .jab file's classifier "
        "structure. A synonym map resolves naming variations between different JAABA "
        "versions and custom classifiers (e.g., 'speed' maps to 'vel', 'bearing' maps "
        "to 'bearing_to_other', 'facingangle' maps to 'facing_angle'). The relative "
        "transform bins values into percentile ranks, following JAABA's convention."
    )

    # ===== SECTION 6: CLASSIFICATION =====
    pdf.add_page()
    pdf.chapter_title("6", "Behavior Classification")

    pdf.section_title("6.1", "Heuristic (Rule-Based) Classifier")
    pdf.body_text(
        "The heuristic classifier implements rule-based detectors for five courtship "
        "behaviors. Each rule computes a per-frame binary score from thresholded feature "
        "comparisons. The thresholds are set to default values derived from published "
        "behavioral criteria and can be adjusted through the Streamlit interface. All "
        "distance-based thresholds are specified in millimeters and converted to pixels "
        "using the user-supplied calibration factor (px_per_mm)."
    )
    pdf.ln(2)

    pdf.subsection_title("Behavior 1: Wing Extension")
    pdf.body_text(
        "Wing extension is the hallmark of the courtship song. The male extends one wing "
        "to approximately 90 degrees from the body axis and vibrates it to produce a "
        "species-specific song composed of pulse and sine components (Greenspan and "
        "Ferveur, 2000)."
    )
    pdf.body_text(
        "Detection rule: A frame is scored positive when the geometric wing angle exceeds "
        "a threshold (default: 60 degrees). The default threshold is set below the full "
        "90-degree extension to capture partial extensions that are still part of the "
        "courtship song. Bouts shorter than 1.0 second are rejected to suppress transient "
        "wing movements unrelated to courtship."
    )

    pdf.subsection_title("Behavior 2: Following")
    pdf.body_text(
        "Following occurs when the male orients toward the female and moves toward her. "
        "It is one of the most common courtship behaviors and often occurs concurrently "
        "with wing extension (Hall, 1994)."
    )
    pdf.body_text(
        "Detection rule: A frame is scored positive when: (1) the forward (parallel) "
        "velocity v_par exceeds 2.0 mm/s; (2) the absolute facing angle toward the "
        "partner is less than 45 degrees; (3) the centroid-to-centroid distance is between "
        "2.0 mm (minimum, to exclude copulation-range proximity) and 8.0 mm (maximum, to "
        "exclude independent locomotion far from the partner). The use of forward velocity "
        "(v_par) rather than total speed is deliberate: it ensures that only motion directed "
        "toward the partner is counted, excluding lateral or backward movements. Bouts "
        "shorter than 1.0 second are rejected."
    )

    pdf.subsection_title("Behavior 3: Circling")
    pdf.body_text(
        "Circling is a lateral movement pattern in which the male moves around the female, "
        "maintaining proximity while oriented at an oblique angle. It typically occurs "
        "during close-range courtship."
    )
    pdf.body_text(
        "Detection rule: A frame is scored positive when: (1) the absolute lateral "
        "(perpendicular) velocity exceeds 2.0 mm/s; (2) the centroid-to-centroid distance "
        "is less than 10.0 mm; (3) the absolute facing angle is between 45 degrees and "
        "135 degrees (indicating the fly is oriented neither directly toward nor directly "
        "away from the partner, consistent with lateral circling). Bouts shorter than 1.0 "
        "second are rejected."
    )

    pdf.subsection_title("Behavior 4: Copulation")
    pdf.body_text(
        "Copulation is characterized by sustained close contact between both flies with "
        "minimal movement. It is typically the longest-duration courtship behavior, lasting "
        "from several minutes up to approximately 20 minutes in D. melanogaster."
    )
    pdf.body_text(
        "Detection rule: A frame is scored positive when: (1) the centroid-to-centroid "
        "distance is less than 2.0 mm; (2) the fly's total velocity is less than 0.3 mm/s; "
        "(3) the rolling standard deviation of the centroid-to-centroid distance over a "
        "2-second window is less than 0.5 mm, ensuring the close contact is spatially "
        "stable rather than a momentary overlap. Bouts shorter than 8.0 seconds are "
        "rejected, consistent with the known minimum duration of copulation."
    )

    pdf.subsection_title("Behavior 5: Attempted Copulation")
    pdf.body_text(
        "Attempted copulation occurs when the male's head makes contact with the female's "
        "body while he is still in motion -- typically involving mounting attempts that "
        "do not result in sustained copulation."
    )
    pdf.body_text(
        "Detection rule: A frame is scored positive when: (1) the nose-to-ellipse distance "
        "(n2e) is less than 0.5 mm (indicating physical contact between the male's head "
        "and the female's body); (2) the total velocity exceeds 1.0 mm/s (distinguishing "
        "active mounting attempts from the static posture of actual copulation). Bouts "
        "shorter than 0.5 seconds are rejected."
    )
    pdf.ln(2)

    pdf.body_text("The complete set of default parameters is summarized in the following table:")
    pdf.ln(2)
    w = [55, 50, 30, 40]
    pdf.table_row(["Behavior", "Parameter", "Default", "Unit"], w, bold=True, fill=True)
    pdf.table_row(["Wing Extension", "Angle threshold", "60", "degrees"], w)
    pdf.table_row(["Wing Extension", "Min duration", "1.0", "seconds"], w)
    pdf.table_row(["Following", "Forward velocity", "2.0", "mm/s"], w)
    pdf.table_row(["Following", "Facing angle", "45", "degrees"], w)
    pdf.table_row(["Following", "Distance range", "2.0-8.0", "mm"], w)
    pdf.table_row(["Following", "Min duration", "1.0", "seconds"], w)
    pdf.table_row(["Circling", "Lateral velocity", "2.0", "mm/s"], w)
    pdf.table_row(["Circling", "Max distance", "10.0", "mm"], w)
    pdf.table_row(["Circling", "Facing angle range", "45-135", "degrees"], w)
    pdf.table_row(["Circling", "Min duration", "1.0", "seconds"], w)
    pdf.table_row(["Copulation", "Max distance", "2.0", "mm"], w)
    pdf.table_row(["Copulation", "Max velocity", "0.3", "mm/s"], w)
    pdf.table_row(["Copulation", "Stability (c2c std)", "0.5", "mm"], w)
    pdf.table_row(["Copulation", "Min duration", "8.0", "seconds"], w)
    pdf.table_row(["Attempted Cop.", "Nose distance", "0.5", "mm"], w)
    pdf.table_row(["Attempted Cop.", "Min velocity", "1.0", "mm/s"], w)
    pdf.table_row(["Attempted Cop.", "Min duration", "0.5", "seconds"], w)
    pdf.table_row(["General", "Merge gap", "0.5", "seconds"], w)

    pdf.section_title("6.2", "JAABA Boosting Classifier")
    pdf.body_text(
        "As an alternative to the heuristic classifier, the system supports JAABA-trained "
        "GentleBoost classifiers (Kabra et al., 2013; Friedman et al., 2000). JAABA models "
        "are stored in .jab files, which are MATLAB v7 or v7.3 (HDF5) archives containing "
        "the trained weak learners (decision stumps) and the feature specification."
    )
    pdf.body_text(
        "Each weak learner is parameterized by four values: (1) dim, the index of the "
        "feature it evaluates (1-based, following MATLAB convention); (2) thr, the threshold; "
        "(3) direction, indicating which side of the threshold corresponds to the positive "
        "class; and (4) alpha, the weight of this learner in the ensemble. The strong "
        "classifier score for each frame is the sum of all weak learner contributions:"
    )
    pdf.ln(2)
    pdf.code_block(
        "score(t) = sum_k [ (2 * I(x[dim_k](t) < thr_k) - 1) * dir_k * alpha_k ]"
    )
    pdf.body_text(
        "where I(.) is the indicator function. Frames with positive total score are "
        "classified as exhibiting the behavior. The .jab parser supports both MATLAB v7 "
        "format (via scipy.io.loadmat) and v7.3 HDF5 format (via h5py), and handles "
        "nested structures, string/byte encoding variations, and sidecar token files for "
        "feature name specification."
    )

    pdf.section_title("6.3", "Temporal Smoothing and Bout Detection")
    pdf.body_text(
        "Raw binary classification scores are temporally smoothed before bout extraction. "
        "A Gaussian filter (sigma = 0.05 * fps frames, typically 1-2 frames at 30 fps) "
        "is applied to the binary score array, and the result is re-thresholded at 0.5. "
        "This eliminates isolated single-frame false positives while preserving the "
        "boundaries of genuine bouts."
    )
    pdf.body_text(
        "Bouts are extracted by identifying contiguous runs of positive scores. Each run "
        "must meet a minimum duration threshold (specific to each behavior, as described "
        "above). After initial extraction, a merge step combines bouts separated by gaps "
        "shorter than a configurable merge threshold (default: 0.5 seconds). This prevents "
        "a single courtship episode from being fragmented into many short bouts by brief "
        "score fluctuations. After merging, the minimum duration filter is applied again "
        "to reject any merged bouts that still fall below the threshold."
    )

    pdf.section_title("6.4", "Mutual Exclusion and Priority Resolution")
    pdf.body_text(
        "Some behaviors are physically incompatible and should not co-occur in the same "
        "frame. The system enforces mutual exclusion through a priority hierarchy:"
    )
    pdf.body_text("Copulation > Attempted Copulation > Following > Circling")
    pdf.body_text(
        "Wing Extension is exempt from mutual exclusion, as it can co-occur with following "
        "or circling (the male often sings while following). For the remaining four "
        "behaviors, frames are claimed in priority order: once a frame is assigned to a "
        "higher-priority behavior, it cannot be claimed by a lower-priority one. Bouts "
        "that lose all their unclaimed frames are removed."
    )

    # ===== SECTION 7: OUTPUT =====
    pdf.add_page()
    pdf.chapter_title("7", "Output and Summary Statistics")

    pdf.section_title("7.1", "Courtship Index")
    pdf.body_text(
        "The Courtship Index (CI) is computed following the definition of Siegel and Hall "
        "(1979): the percentage of the total observation time during which the fly engages "
        "in any courtship behavior. The implementation creates a boolean array spanning all "
        "frames, marks frames belonging to any detected bout (across all five behaviors "
        "and all flies), and computes CI = (marked frames / total frames) * 100. The CI "
        "is reported separately for each fly."
    )

    pdf.section_title("7.2", "Sex Inference")
    pdf.body_text(
        "When sex labels are not provided a priori, the system infers sex from behavioral "
        "asymmetry. The fly that accounts for more than 70% of the total wing extension "
        "duration is classified as male, and the other as female. This ratio-based criterion "
        "reflects the strong sexual dimorphism in courtship song production: males perform "
        "the vast majority of wing extensions in a typical courtship assay. If neither fly "
        "exceeds the 70% threshold (as may occur when both flies are male or when courtship "
        "is minimal), no sex assignment is made."
    )

    pdf.section_title("7.3", "CSV Export Format")
    pdf.body_text("Two CSV files are produced per chamber:")
    pdf.body_text(
        "Detailed results (chamber_N_results.csv): One row per behavior bout, with columns "
        "for Behavior, Fly, Start_Time (s), End_Time (s), Duration (s), Start_Frame, and "
        "End_Frame. Rows are sorted chronologically."
    )
    pdf.body_text(
        "Summary statistics (chamber_N_summary.csv): Per-fly metrics in a long-format table "
        "with columns Fly, Metric, and Value. Metrics include per-behavior bout counts and "
        "durations, Courtship Index (%), Latency to First Courtship (s), Latency to "
        "Copulation (s), and Video Duration (s)."
    )

    # ===== SECTION 8: INSTALLATION =====
    pdf.add_page()
    pdf.chapter_title("8", "Installation and Usage")

    pdf.section_title("8.1", "System Requirements")
    pdf.bullet_point("Python 3.9 or later")
    pdf.bullet_point("A video file of a multi-chamber courtship arena (MP4, AVI, MOV, or MKV)")
    pdf.bullet_point("Sufficient RAM for parallel chamber processing (approximately 0.5 GB per chamber)")
    pdf.bullet_point("Optional: FFmpeg installed and available on PATH for faster video cropping")
    pdf.bullet_point("Optional: JAABA .jab classifier files for boosting-based classification")

    pdf.section_title("8.2", "Installation Steps")
    pdf.body_text("1. Clone or download the project repository:")
    pdf.code_block("git clone <repository-url>\ncd Courtship_Analysis_Python")
    pdf.body_text("2. Create and activate a virtual environment (recommended):")
    pdf.code_block(
        "python -m venv venv\n"
        "# On Windows:\n"
        "venv\\Scripts\\activate\n"
        "# On macOS/Linux:\n"
        "source venv/bin/activate"
    )
    pdf.body_text("3. Install the required dependencies:")
    pdf.code_block("pip install -r requirements.txt")
    pdf.body_text("The requirements.txt file specifies the following packages:")
    pdf.code_block(
        "streamlit>=1.30.0\n"
        "opencv-python-headless\n"
        "numpy>=1.24.0\n"
        "scipy>=1.10.0\n"
        "pandas>=2.0.0\n"
        "ffmpeg-python>=0.2.0\n"
        "h5py>=3.10.0\n"
        "matplotlib>=3.7.0\n"
        "Pillow>=9.0.0\n"
        "psutil>=5.9.0"
    )

    pdf.section_title("8.3", "Running the Application")
    pdf.body_text("Launch the Streamlit web application:")
    pdf.code_block("streamlit run app.py")
    pdf.body_text(
        "This will start a local web server (typically at http://localhost:8501) and "
        "open the application in a browser. The interface consists of two tabs:"
    )
    pdf.bullet_point(
        "Analysis tab: Upload or specify a video file, detect chambers, adjust parameters, "
        "select chambers, and run the full analysis pipeline."
    )
    pdf.bullet_point(
        "Verification tab: Browse completed analysis results, view GIF previews of detected "
        "bouts, inspect behavior timelines, and review summary statistics."
    )

    pdf.section_title("8.4", "Analysis Workflow")
    pdf.body_text(
        "Step 1: Provide the input video via file upload or by entering a local file path."
    )
    pdf.body_text(
        "Step 2: Review the auto-detected chambers overlaid on the first frame. Adjust using "
        "the Manual Grid Builder or the raw ROI editor if needed."
    )
    pdf.body_text(
        "Step 3: Set the calibration factor (pixels per mm) based on a known dimension in "
        "the arena. This is required for the heuristic classifier's distance-based thresholds."
    )
    pdf.body_text(
        "Step 4: Choose the analysis method (Heuristic or JAABA Classifiers). If using "
        "heuristic mode, threshold parameters can be adjusted in the sidebar expander."
    )
    pdf.body_text(
        "Step 5: Select which chambers to process and whether to enable video stabilization."
    )
    pdf.body_text(
        "Step 6: Click 'Start Analysis.' Progress is displayed in real time, and results "
        "appear incrementally as chambers complete. Processing is parallelized across "
        "available CPU cores (with RAM-aware worker count adjustment)."
    )
    pdf.body_text(
        "Step 7: Download individual CSV files or a ZIP archive of all results."
    )

    pdf.section_title("8.5", "Resumable Processing")
    pdf.body_text(
        "The system saves progress after each chamber completes. If the analysis is "
        "interrupted (browser closure, system crash), re-running on the same video will "
        "detect previously completed chambers and skip them, processing only the remaining "
        "chambers."
    )

    # ===== SECTION 9: REFERENCES =====
    pdf.add_page()
    pdf.chapter_title("9", "References")

    refs = [
        "Branson, K., Robie, A. A., Bender, J., Perona, P., & Dickinson, M. H. (2009). "
        "High-throughput ethomics in large groups of Drosophila. Nature Methods, 6(6), "
        "451-457. doi:10.1038/nmeth.1328",

        "Evangelidis, G. D., & Psarakis, E. Z. (2008). Parametric image alignment using "
        "enhanced correlation coefficient maximization. IEEE Transactions on Pattern "
        "Analysis and Machine Intelligence, 30(10), 1858-1865. doi:10.1109/TPAMI.2008.113",

        "Eyjolfsdottir, E., Branson, S., Burgos-Artizzu, X. P., Hoopfer, E. D., Schor, J., "
        "Anderson, D. J., & Perona, P. (2014). Detecting social actions of fruit flies. "
        "In Computer Vision -- ECCV 2014 (pp. 772-787). Springer. "
        "doi:10.1007/978-3-319-10605-2_50",

        "Fitzgibbon, A. W., Pilu, M., & Fisher, R. B. (1999). Direct least square fitting "
        "of ellipses. IEEE Transactions on Pattern Analysis and Machine Intelligence, "
        "21(5), 476-480. doi:10.1109/34.765658",

        "Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: "
        "A statistical view of boosting. The Annals of Statistics, 28(2), 337-407. "
        "doi:10.1214/aos/1016218223",

        "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. "
        "The Annals of Statistics, 29(5), 1189-1232. doi:10.1214/aos/1013203451",

        "Greenspan, R. J., & Ferveur, J.-F. (2000). Courtship in Drosophila. Annual Review "
        "of Genetics, 34, 205-232. doi:10.1146/annurev.genet.34.1.205",

        "Hall, J. C. (1994). The mating of a fly. Science, 264(5166), 1702-1714. "
        "doi:10.1126/science.8209251",

        "Kabra, M., Robie, A. A., Rivera-Alba, M., Branson, S., & Branson, K. (2013). "
        "JAABA: Interactive machine learning for automatic annotation of animal behavior. "
        "Nature Methods, 10(1), 64-67. doi:10.1038/nmeth.2281",

        "Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval "
        "Research Logistics Quarterly, 2(1-2), 83-97. doi:10.1002/nav.3800020109",

        "Munkres, J. (1957). Algorithms for the assignment and transportation problems. "
        "Journal of the Society for Industrial and Applied Mathematics, 5(1), 32-38. "
        "doi:10.1137/0105003",

        "Oppenheim, A. V., & Schafer, R. W. (2009). Discrete-Time Signal Processing "
        "(3rd ed.). Prentice Hall. ISBN: 978-0131988422",

        "Siegel, R. W., & Hall, J. C. (1979). Conditioned responses in courtship behavior "
        "of normal and mutant Drosophila. Proceedings of the National Academy of Sciences, "
        "76(7), 3430-3434. doi:10.1073/pnas.76.7.3430",

        "Yamamoto, D., & Koganezawa, M. (2013). Genes and circuits of courtship behaviour "
        "in Drosophila males. Nature Reviews Neuroscience, 14(10), 681-692. "
        "doi:10.1038/nrn3567",

        "Zivkovic, Z. (2004). Improved adaptive Gaussian mixture model for background "
        "subtraction. In Proceedings of the 17th IEEE International Conference on Pattern "
        "Recognition (ICPR) (pp. 28-31). doi:10.1109/ICPR.2004.1333992",

        "Zivkovic, Z., & van der Heijden, F. (2006). Efficient adaptive density estimation "
        "per image pixel for the task of background subtraction. Pattern Recognition "
        "Letters, 27(7), 773-780. doi:10.1016/j.patrec.2005.11.005",
    ]

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 30, 30)
    for i, ref in enumerate(refs, 1):
        pdf.multi_cell(0, 5, f"[{i}]  {ref}")
        pdf.ln(3)

    # ===== OUTPUT =====
    output_path = os.path.join(os.path.dirname(__file__), "Courtship_Analysis_Technical_Documentation.pdf")
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
