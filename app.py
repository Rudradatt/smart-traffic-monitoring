"""
Smart Traffic Monitoring and Vehicle Classification System.
Professional Streamlit Dashboard featuring:
1. Overview: System status, active models, hardware details, recent traffic sessions.
2. Dataset Explorer: Validation, corrupt image detection, class imbalance, synthetic sample generator.
3. Train Model: Custom CNN, ResNet-18, MobileNetV3 training with live curves and metrics.
4. Model Comparison: Side-by-side benchmark of accuracy, F1, latency, throughput, params.
5. Image Classification: Single vehicle crop inference with probability breakdown & uncertainty flags.
6. Live/Video Monitoring: YOLO + ByteTrack vehicle tracking, virtual line counting, density estimation.
7. Traffic Analytics: Historical session insights, time-series density trends, class proportions.
8. Reports: Exportable CSV, JSON, and annotated video artifacts.
9. Settings/About: CUDA verification, COCO mapping limits, dataset sources manifest, speed caveats.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from classifier_models import build_classifier, load_checkpoint
from config import (
    CLASS_COLORS_HEX,
    CLASSIFIER_ARCHITECTURES,
    COCO_VEHICLE_CLASS_IDS,
    DATASET_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DENSITY_THRESHOLDS,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    MODELS_DIR,
    RESULTS_DIR,
    VEHICLE_CLASSES,
    get_device,
)
from database import (
    get_experiments,
    get_latest_experiment,
    get_model_comparisons,
    get_traffic_session_logs,
    get_traffic_sessions,
    init_db,
)
from dataset_utils import (
    create_synthetic_sample_dataset,
    validate_dataset,
)
from detection import VehicleDetector, generate_yolo_dataset_yaml
from evaluate_classifier import evaluate_classifier
from predict_image import predict_vehicle_image
from tracking import VehicleTracker
from traffic_density import DENSITY_DISCLAIMER, TrafficDensityEstimator
from train_classifier import train_classifier
from utils import (
    calculate_model_parameters,
    format_time,
    get_device_info,
    get_file_size_mb,
    setup_logger,
)

logger = setup_logger("streamlit_app")

# Page Configuration
st.set_page_config(
    page_title="Smart Traffic AI & Vehicle Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Navy Theme & Custom CSS
st.markdown(
    """
    <style>
    /* Dark Navy Palette */
    .stApp {
        background-color: #0B111E;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #131E32 0%, #17243B 100%);
        border: 1px solid #1E2E4B;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    /* Badges */
    .badge-green {
        background-color: #064E3B;
        color: #34D399;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-yellow {
        background-color: #78350F;
        color: #FBBF24;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-red {
        background-color: #7F1D1D;
        color: #F87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-blue {
        background-color: #1E3A8A;
        color: #60A5FA;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        border-bottom: 2px solid #1E2E4B;
        padding-bottom: 8px;
        margin-top: 15px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure Database is initialized
init_db()
dev_info = get_device_info()

# Sidebar Navigation
st.sidebar.markdown("### 🚦 Smart Traffic AI")
st.sidebar.markdown(
    f"<span class='badge-blue'>PyTorch {dev_info['pytorch_version']}</span> "
    f"<span class='{'badge-green' if dev_info['cuda_available'] else 'badge-yellow'}'>"
    f"{dev_info['device_name']}</span>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGES = [
    "Overview",
    "Dataset Explorer",
    "Train Model",
    "Model Comparison",
    "Image Classification",
    "Live/Video Monitoring",
    "Traffic Analytics",
    "Reports & Exports",
    "Settings & About",
]

page_choice = st.sidebar.radio("Navigate System", PAGES)

# Discover Available Trained Checkpoints
available_checkpoints = sorted(list(MODELS_DIR.glob("*.pth")))
checkpoint_options = {cp.name: cp for cp in available_checkpoints}

if checkpoint_options:
    selected_cp_name = st.sidebar.selectbox(
        "Active Classifier Checkpoint",
        list(checkpoint_options.keys()),
        index=0,
    )
    active_checkpoint_path = checkpoint_options[selected_cp_name]
else:
    st.sidebar.info("No trained checkpoints yet in `models/`.")
    active_checkpoint_path = None

st.sidebar.markdown("---")
st.sidebar.caption("Smart Traffic & Vehicle Classifier v1.0.0\nFully reproducible & verified.")


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page_choice == "Overview":
    st.title("Traffic Monitoring & Vehicle Classification System")
    st.markdown(
        "Deep Learning surveillance system integrating a custom scratch-trained CNN, "
        "transfer learning benchmarks (ResNet-18, MobileNetV3-Small), Ultralytics YOLO object detection, "
        "ByteTrack multi-object tracking, and virtual line-crossing density analytics."
    )

    # Top Metric Tiles
    experiments = get_experiments()
    sessions = get_traffic_sessions()
    latest_exp = get_latest_experiment()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Compute Hardware</div>
                <div class='metric-value'>{dev_info['current_device'].upper()}</div>
                <div class='metric-sub'>{dev_info['device_name']} ({dev_info.get('total_vram_gb', 0)} GB)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Vehicle Classes</div>
                <div class='metric-value'>{len(VEHICLE_CLASSES)}</div>
                <div class='metric-sub'>Auto, Rickshaw, 2-Wheelers, Bus, Truck, Car</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Logged Experiments</div>
                <div class='metric-value'>{len(experiments)}</div>
                <div class='metric-sub'>Recorded in SQLite database</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-title'>Monitored Sessions</div>
                <div class='metric-value'>{len(sessions)}</div>
                <div class='metric-sub'>Video traffic runs stored</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>Latest Experiment Snapshot</div>", unsafe_allow_html=True)
    if latest_exp:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Architecture", latest_exp["architecture"])
        e1.caption(f"Status: {latest_exp['status']}")
        e2.metric("Best Val Accuracy", f"{latest_exp['best_val_accuracy']*100:.2f}%" if latest_exp['best_val_accuracy'] else "N/A")
        e2.caption(f"Val Loss: {latest_exp['best_val_loss']:.4f}" if latest_exp['best_val_loss'] else "")
        e3.metric("Test Accuracy", f"{latest_exp['test_accuracy']*100:.2f}%" if latest_exp['test_accuracy'] else "Not Evaluated")
        e3.caption(f"Test F1: {latest_exp['test_macro_f1']*100:.2f}%" if latest_exp['test_macro_f1'] else "")
        e4.metric("Epochs Completed", f"{latest_exp['epochs']}")
        e4.caption(f"Duration: {format_time(latest_exp['training_duration_seconds'] or 0)}")
    else:
        st.info("No training runs recorded yet. Head over to **Train Model** to run your first experiment!")

    # System Architecture Summary
    st.markdown("<div class='section-header'>Architectural Pipeline</div>", unsafe_allow_html=True)
    st.markdown(
        """
        - **1. Data Preprocessing & Validation**: Stratified train/val/test splits, corrupt image filtering, class imbalance compensation via inverse-frequency weights.
        - **2. Classifier Family**:
            - `CustomTrafficCNN`: 4-stage convolutional feature extractor with BatchNorm, ReLU, AdaptiveAvgPool, and Dropout.
            - `ResNet18Transfer`: Deep residual baseline with pretrained weights and custom linear head.
            - `MobileNetV3Transfer`: Efficient inverted bottleneck architecture optimized for edge latency.
        - **3. Object Detection & Tracking**: Ultralytics YOLOv8/11 coupled with ByteTrack to track vehicles and assign persistent track IDs.
        - **4. Traffic Counting & Density**: Line crossing vector math preventing double counts, plus optical occupancy ratio for Low/Med/High classification.
        """
    )


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif page_choice == "Dataset Explorer":
    st.title("Dataset Explorer & Validation Engine")
    st.markdown(
        "Inspect dataset integrity, verify folder structure, detect corrupt images, "
        "and analyze class imbalance prior to model training."
    )

    dataset_path_str = st.text_input("Dataset Directory Path", value=str(DATASET_DIR))
    c_btn1, c_btn2 = st.columns(2)
    run_val = c_btn1.button("🔍 Run Dataset Validation", use_container_width=True)
    gen_sample = c_btn2.button("🧪 Generate Sample Synthetic Dataset (7-Class)", use_container_width=True)

    if gen_sample:
        with st.spinner("Generating sample synthetic dataset with 7 vehicle classes..."):
            out_p = create_synthetic_sample_dataset(dataset_path_str, samples_per_class=14)
            st.success(f"Sample dataset generated at `{out_p}`! You can now validate or train immediately.")

    if run_val or Path(dataset_path_str).exists():
        report = validate_dataset(dataset_path_str)

        if report["is_valid"]:
            st.success(f"Dataset is valid! Found {report['total_valid_images']} total images.")
        else:
            st.error(f"Dataset validation encountered issues: {', '.join(report['errors'])}")

        if report["warnings"]:
            for w in report["warnings"]:
                st.warning(w)

        # Split Summary Tiles
        st.markdown("<div class='section-header'>Partition Counts</div>", unsafe_allow_html=True)
        cols = st.columns(len(report["splits_found"]) if report["splits_found"] else 1)
        for idx, split in enumerate(report["splits_found"]):
            with cols[idx]:
                cnt = sum(report["counts_by_split"].get(split, {}).values())
                st.metric(f"Split: {split.upper()}", f"{cnt} images")

        # Class Distribution Chart
        if report["counts_by_split"]:
            st.markdown("<div class='section-header'>Class Distribution</div>", unsafe_allow_html=True)
            df_counts = pd.DataFrame(report["counts_by_split"]).fillna(0).astype(int)
            st.bar_chart(df_counts)

        # Imbalance Analysis
        if report.get("class_imbalance"):
            imb = report["class_imbalance"]
            st.markdown("<div class='section-header'>Class Imbalance Assessment</div>", unsafe_allow_html=True)
            i1, i2, i3 = st.columns(3)
            i1.metric("Imbalance Ratio (Max / Min)", f"{imb['imbalance_ratio']}x")
            i2.metric("Severity", imb["severity"])
            i3.metric("Recommended Action", "Inverse Class Weighting Enabled" if imb["imbalance_ratio"] > 1.5 else "Standard Loss")


# ==============================================================================
# PAGE 3: TRAIN MODEL
# ==============================================================================
elif page_choice == "Train Model":
    st.title("Train Vehicle Classifier")
    st.markdown(
        "Train a custom CNN from scratch or fine-tune ResNet-18 / MobileNetV3-Small. "
        "Class count is dynamically inferred from your dataset."
    )

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        arch_select = st.selectbox("Model Architecture", CLASSIFIER_ARCHITECTURES, index=0)
        epochs_select = st.slider("Epochs", min_value=1, max_value=50, value=5)
        batch_select = st.select_slider("Batch Size", options=[4, 8, 16, 32, 64], value=16)
        img_size_select = st.select_slider("Image Resolution (px)", options=[128, 224, 256], value=224)

    with col_cfg2:
        lr_select = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        weight_decay_select = st.number_input("Weight Decay", value=DEFAULT_WEIGHT_DECAY, format="%.5f")
        patience_select = st.number_input("Early Stopping Patience", min_value=1, max_value=20, value=DEFAULT_PATIENCE)
        use_weights = st.checkbox("Apply Class Imbalance Weights", value=True)

    data_dir_input = st.text_input("Dataset Directory Path", value=str(DATASET_DIR))

    if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
        p_bar = st.progress(0)
        status_txt = st.empty()
        metrics_table = st.empty()
        plot_placeholder = st.empty()

        epoch_logs: List[Dict[str, Any]] = []

        def on_epoch_update(curr_epoch: int, total_epochs: int, record: Dict[str, float]):
            p_bar.progress(curr_epoch / total_epochs)
            status_txt.markdown(
                f"**Epoch {curr_epoch}/{total_epochs}** | "
                f"Train Loss: `{record['train_loss']}` | Train Acc: `{record['train_acc']*100:.1f}%` | "
                f"Val Loss: `{record['val_loss']}` | Val Acc: `{record['val_acc']*100:.1f}%`"
            )
            epoch_logs.append(record)
            metrics_table.dataframe(pd.DataFrame(epoch_logs).tail(5), use_container_width=True)

        try:
            with st.spinner("Training in progress..."):
                train_res = train_classifier(
                    data_dir=Path(data_dir_input),
                    arch=arch_select,
                    epochs=epochs_select,
                    batch_size=batch_select,
                    lr=lr_select,
                    weight_decay=weight_decay_select,
                    image_size=img_size_select,
                    device_name="auto",
                    patience=patience_select,
                    use_class_weights=use_weights,
                    save_dir=MODELS_DIR,
                    results_dir=RESULTS_DIR,
                    progress_callback=on_epoch_update,
                )
            st.success(f"Training Complete! Saved checkpoint: `{train_res['checkpoint_path']}`")
            if Path(train_res["curves_png"]).exists():
                st.image(train_res["curves_png"], caption="Training and Validation Loss/Accuracy Curves")
        except Exception as e:
            st.error(f"Training Failed: {e}")


# ==============================================================================
# PAGE 4: MODEL COMPARISON
# ==============================================================================
elif page_choice == "Model Comparison":
    st.title("Model Comparison & Benchmark")
    st.markdown(
        "Evaluate multiple trained classifiers side-by-side on the identical test split. "
        "Benchmark parameter count, checkpoint size, latency, throughput, and macro F1."
    )

    all_cps = sorted(list(MODELS_DIR.glob("*.pth")))
    if not all_cps:
        st.warning("No `.pth` checkpoints found in `models/`. Please train models first in **Train Model**.")
    else:
        selected_cps = st.multiselect(
            "Select Checkpoints to Compare",
            [c.name for c in all_cps],
            default=[c.name for c in all_cps][:3],
        )
        test_dir_in = st.text_input("Held-out Test Directory", value=str(DATASET_DIR / "test"))

        if st.button("⚖️ Run Benchmark Comparison", type="primary") and selected_cps:
            with st.spinner("Running benchmarks and computing metrics..."):
                from compare_models import compare_models
                cp_paths = [MODELS_DIR / name for name in selected_cps]
                try:
                    comp_res = compare_models(
                        checkpoints=cp_paths,
                        test_dir=test_dir_in,
                        output_dir=RESULTS_DIR,
                        device_name="auto",
                    )
                    st.success("Benchmark completed!")
                    df_comp = pd.DataFrame(comp_res["comparison_table"])
                    st.dataframe(df_comp, use_container_width=True)

                    if Path(comp_res["comparison_plot"]).exists():
                        st.image(comp_res["comparison_plot"], caption="Comparative Metrics & Latency")
                except Exception as e:
                    st.error(f"Comparison benchmark failed: {e}")


# ==============================================================================
# PAGE 5: IMAGE CLASSIFICATION
# ==============================================================================
elif page_choice == "Image Classification":
    st.title("Single Vehicle Image Inference")
    st.markdown(
        "Classify a single vehicle crop with softmax probability distribution and uncertainty detection."
    )

    if not active_checkpoint_path:
        st.warning("Please train or provide a model checkpoint in `models/`.")
    else:
        st.info(f"Using Checkpoint: `{active_checkpoint_path.name}`")
        conf_thresh = st.slider("Confidence Certainty Threshold", 0.1, 1.0, 0.50, 0.05)

        img_file = st.file_uploader("Upload Vehicle Image Crop", type=["jpg", "jpeg", "png", "webp"])

        if img_file is not None:
            pil_image = Image.open(img_file).convert("RGB")
            c_img, c_res = st.columns([1, 1])

            with c_img:
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)

            with c_res:
                pred = predict_vehicle_image(
                    image_input=pil_image,
                    checkpoint_path=active_checkpoint_path,
                    confidence_threshold=conf_thresh,
                    device_name="auto",
                )

                if pred["is_uncertain"]:
                    st.markdown(f"<span class='badge-yellow'>⚠️ {pred['uncertainty_warning']}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-green'>Predicted: {pred['predicted_class'].upper()}</span>", unsafe_allow_html=True)

                st.metric("Confidence", f"{pred['confidence_percentage']}%")
                st.metric("Inference Latency", f"{pred['inference_latency_ms']} ms", f"on {pred['device'].upper()}")

                st.markdown("**Probability Distribution:**")
                df_p = pd.DataFrame(list(pred["class_probabilities"].items()), columns=["Class", "Probability"])
                st.bar_chart(df_p.set_index("Class"))

            st.caption(f"ℹ️ {pred['educational_notice']}")


# ==============================================================================
# PAGE 6: LIVE / VIDEO MONITORING
# ==============================================================================
elif page_choice == "Live/Video Monitoring":
    st.title("Live / Video Traffic Surveillance & Counting")
    st.markdown(
        "Detect, track with ByteTrack, count bidirectional line crossings, "
        "and estimate instantaneous traffic density."
    )

    # Sidebar settings for video
    st.markdown("<div class='section-header'>Detection & Tracking Configuration</div>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        yolo_weight = st.text_input("YOLO Model Weight", value="yolov8n.pt")
        conf_cut = st.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    with v2:
        use_refiner = st.checkbox("Enable CNN Subtype Refinement", value=bool(active_checkpoint_path))
        refiner_cp = active_checkpoint_path if use_refiner else None
        line_y_ratio = st.slider("Counting Line Y-Position (%)", 20, 80, 55, 5)
    with v3:
        low_max_thresh = st.number_input("Low Density Max (vehicles)", value=4)
        med_max_thresh = st.number_input("Medium Density Max (vehicles)", value=10)

    video_file = st.file_uploader("Upload Road Traffic Video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save temporary video file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.write(f"Video Resolution: `{width}x{height}` | Total Frames: `{total_v_frames}` | FPS: `{orig_fps:.1f}`")

        # Initialize Tracker & Density Estimator
        counting_y = int(height * (line_y_ratio / 100.0))
        counting_line = ((20, counting_y), (width - 20, counting_y))

        tracker = VehicleTracker(
            model_path=yolo_weight,
            tracker_type="bytetrack.yaml",
            confidence_threshold=conf_cut,
            refiner_checkpoint=refiner_cp,
        )
        density_est = TrafficDensityEstimator(
            counting_line=counting_line,
            density_thresholds={"low_max": low_max_thresh, "medium_max": med_max_thresh},
        )

        start_proc = st.button("▶️ Process Traffic Video", type="primary", use_container_width=True)

        if start_proc:
            frame_window = st.image([])
            stat_window = st.empty()
            p_bar = st.progress(0)

            # Output video writer
            out_video_path = RESULTS_DIR / "annotated_traffic_output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(out_video_path), fourcc, orig_fps, (width, height))

            frame_idx = 0
            session_frame_logs = []
            start_clock = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                elapsed_sec = frame_idx / orig_fps

                # 1. Tracking update
                tracks = tracker.update(frame)

                # 2. Density & Counting update
                metrics = density_est.update(tracks, frame.shape, elapsed_seconds=elapsed_sec)

                # 3. Draw annotations
                annotated = tracker.draw_tracks(frame, tracks, draw_trails=True)
                annotated = density_est.draw_density_overlay(annotated, metrics, show_hud=True)

                # Write frame to output video
                out_writer.write(annotated)

                # Record frame log
                session_frame_logs.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(elapsed_sec, 2),
                        "vehicle_count": metrics["active_vehicles"],
                        "density_level": metrics["density_level"],
                        "density_ratio": metrics["occupancy_ratio"],
                        "cumulative_counts": metrics["class_counts"],
                    }
                )

                # Update preview at periodic intervals
                if frame_idx % 2 == 0 or frame_idx == total_v_frames:
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_window.image(rgb_frame, use_container_width=True)
                    if total_v_frames > 0:
                        p_bar.progress(min(1.0, frame_idx / total_v_frames))

                    # Live status banner
                    stat_window.markdown(
                        f"**Frame:** `{frame_idx}/{total_v_frames}` | "
                        f"**Density:** `{metrics['density_level']}` | "
                        f"**Total Passed:** `{metrics['total_counted']}` | "
                        f"**Rate:** `{metrics['vehicles_per_minute']} VPM`"
                    )

            cap.release()
            out_writer.release()

            st.success("Video processing complete!")

            # Summarize and record session to database
            from analytics import summarize_traffic_session
            sess_id, sess_sum = summarize_traffic_session(
                frame_logs=session_frame_logs,
                session_name=f"Traffic_Session_{int(time.time())}",
                video_source=video_file.name,
                fps=orig_fps,
                detector_used=yolo_weight,
                tracker_used="ByteTrack",
                classifier_refinement_used=bool(refiner_cp),
            )

            st.info(f"Traffic session recorded as Session ID #{sess_id} in SQLite database.")
            st.markdown(f"💾 Annotated video saved to: `{out_video_path}`")
    else:
        st.info("Upload an MP4/AVI/MOV traffic video to start live surveillance.")


# ==============================================================================
# PAGE 7: TRAFFIC ANALYTICS
# ==============================================================================
elif page_choice == "Traffic Analytics":
    st.title("Traffic Analytics & Historical Sessions")
    st.markdown("Analyze vehicle counts, congestion profiles, and class proportions across sessions.")

    sessions = get_traffic_sessions()
    if not sessions:
        st.info("No recorded sessions found in the database. Process a video in **Live/Video Monitoring** first.")
    else:
        session_names = {f"Session #{s['id']} - {s['session_name']} ({s['timestamp'][:10]})": s["id"] for s in sessions}
        picked_name = st.selectbox("Select Traffic Session", list(session_names.keys()))
        picked_id = session_names[picked_name]

        logs = get_traffic_session_logs(picked_id)

        if logs:
            df_logs = pd.DataFrame(logs)

            a1, a2, a3, a4 = st.columns(4)
            s_meta = next(s for s in sessions if s["id"] == picked_id)
            a1.metric("Total Counted", s_meta.get("total_vehicles_counted", 0))
            a2.metric("Peak Density", s_meta.get("peak_density_level", "Low"))
            a3.metric("Duration", f"{s_meta.get('duration_seconds', 0):.1f}s")
            a4.metric("Avg Vehicles / Frame", f"{s_meta.get('avg_density_count', 0):.1f}")

            st.markdown("<div class='section-header'>Vehicle Congestion Profile Over Time</div>", unsafe_allow_html=True)
            st.line_chart(df_logs.set_index("timestamp_sec")["vehicle_count"])

            st.markdown("<div class='section-header'>Cumulative Vehicle Class Breakdown</div>", unsafe_allow_html=True)
            last_counts = s_meta.get("class_counts", {})
            if last_counts:
                df_cls = pd.DataFrame(list(last_counts.items()), columns=["Vehicle Class", "Count"])
                st.bar_chart(df_cls.set_index("Vehicle Class"))
            else:
                st.caption("No class-specific counts logged.")


# ==============================================================================
# PAGE 8: REPORTS & EXPORTS
# ==============================================================================
elif page_choice == "Reports & Exports":
    st.title("Experiment & Traffic Reports")
    st.markdown("Download machine-readable JSON and CSV reports and annotated video artifacts.")

    tab1, tab2 = st.tabs(["Model Training Experiments", "Traffic Surveillance Sessions"])

    with tab1:
        experiments = get_experiments()
        if experiments:
            df_exp = pd.DataFrame(experiments)
            st.dataframe(df_exp, use_container_width=True)

            csv_data = df_exp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Experiments CSV",
                csv_data,
                "experiments_report.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("No experiment logs found.")

    with tab2:
        sessions = get_traffic_sessions()
        if sessions:
            df_sess = pd.DataFrame(sessions)
            st.dataframe(df_sess, use_container_width=True)

            csv_sess = df_sess.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Traffic Sessions CSV",
                csv_sess,
                "traffic_sessions_report.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("No traffic sessions found.")


# ==============================================================================
# PAGE 9: SETTINGS & ABOUT
# ==============================================================================
elif page_choice == "Settings & About":
    st.title("System Settings & Research Documentation")

    st.markdown("<div class='section-header'>Hardware & Device Diagnostics</div>", unsafe_allow_html=True)
    st.json(dev_info)

    st.markdown("<div class='section-header'>Standard COCO to 7-Class Vehicle Mapping</div>", unsafe_allow_html=True)
    st.table(
        pd.DataFrame(
            [
                {"Target Class": "car", "COCO Native Label": "car (id: 2)", "Sub-type Disambiguation": "Native"},
                {"Target Class": "motorcycle", "COCO Native Label": "motorcycle (id: 3)", "Sub-type Disambiguation": "Lumped with scooter"},
                {"Target Class": "bus", "COCO Native Label": "bus (id: 5)", "Sub-type Disambiguation": "Native"},
                {"Target Class": "truck", "COCO Native Label": "truck (id: 7)", "Sub-type Disambiguation": "Native"},
                {"Target Class": "bicycle", "COCO Native Label": "bicycle (id: 1)", "Sub-type Disambiguation": "Native"},
                {"Target Class": "auto_rickshaw", "COCO Native Label": "None (Missing in COCO)", "Sub-type Disambiguation": "Requires CNN Refiner / Custom YOLO"},
                {"Target Class": "scooter", "COCO Native Label": "None (Mapped to motorcycle)", "Sub-type Disambiguation": "Requires CNN Refiner / Custom YOLO"},
            ]
        )
    )

    st.markdown("<div class='section-header'>Dataset Manifest & Public Sources</div>", unsafe_allow_html=True)
    st.markdown(
        """
        | Dataset Name | Primary Vehicle Classes | License | Limitation / Notes |
        |---|---|---|---|
        | **Stanford Cars** | Cars (196 sub-models) | Research / CC BY | Cars only; no two-wheelers or auto-rickshaws |
        | **Indian Driving Dataset (IDD)** | Auto-rickshaw, Motorcycle, Bus, Car, Truck, Bicycle | CC BY-NC 4.0 | Excellent coverage for Indian road conditions |
        | **BDD100K** | Car, Bus, Truck, Motorcycle, Bicycle | BSD-3-Clause | Large diversity; lacks auto-rickshaw |
        | **MIO-TCD** | Motorized & non-motorized classes | Non-commercial | Traffic camera viewpoints |
        """
    )

    st.markdown("<div class='section-header'>Scientific Rigor & Limitations</div>", unsafe_allow_html=True)
    st.warning(DENSITY_DISCLAIMER)
    st.markdown(
        """
        - **Speed Estimation**: True speed calculation (km/h) requires camera intrinsic and extrinsic matrix calibration to project 2D image coordinates onto the ground plane. Estimating speeds without calibration markers produces scientifically arbitrary values.
        - **Classifier vs. Detector**: A classifier processes isolated vehicle image crops. To locate vehicles in full road scenes, an object detector (YOLO) is required.
        """
    )
