"""
Smart Traffic Monitor - Streamlit dashboard.

Pages (sidebar label -> internal page key):
    Dashboard        -> Overview
    Dataset Check    -> Dataset Explorer
    Train            -> Train Model
    Compare Models   -> Model Comparison
    Classify Image   -> Image Classification
    Video Counting   -> Live/Video Monitoring
    Traffic Insights -> Traffic Analytics
    Exports          -> Reports & Exports
    System & Notes   -> Settings & About

Only the presentation layer changed. Every backend call (training, detection,
tracking, density, database, prediction) is used exactly as before.
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

# ------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Traffic Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Design system
#   Asphalt sidebar, concrete workspace, lane-marking yellow as the signature.
#   Green = actions/healthy, amber = caution, red = problems.
#   Type: Barlow Semi Condensed (headings, numbers) + Barlow (body).
# ------------------------------------------------------------------------------
ASPHALT = "#1F2326"
CONCRETE = "#ECEDEA"
PAINT = "#F5B700"
SIGNAL = "#1F7A5C"
BRAKE = "#C8372D"
INK = "#15181A"
MUTED = "#677074"

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap');

:root {
    --asphalt: #1F2326;
    --concrete: #ECEDEA;
    --paint: #F5B700;
    --signal: #1F7A5C;
    --brake: #C8372D;
    --ink: #15181A;
    --muted: #677074;
    --line: #D6D9D3;
    --radius: 4px;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Barlow', 'Segoe UI', sans-serif;
}
.stApp { background: var(--concrete); color: var(--ink); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding-top: 2.2rem; padding-bottom: 4rem; }

/* ---------- Page header ---------- */
.page-head { margin-bottom: 1.6rem; }
.page-head h1 {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 2.3rem; font-weight: 700; letter-spacing: -0.01em;
    color: var(--ink); margin: 0; padding: 0; line-height: 1.1;
}
.page-head p { color: var(--muted); font-size: 1.02rem; margin: 6px 0 0 0; max-width: 70ch; }
.lane {
    height: 4px; margin-top: 16px;
    background: repeating-linear-gradient(90deg, var(--paint) 0 28px, transparent 28px 46px);
}

/* ---------- Section headers ---------- */
.section-header {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 1.3rem; font-weight: 600; color: var(--ink);
    margin: 2rem 0 0.9rem 0; padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
}

/* ---------- Stat panels ---------- */
.panel {
    background: #fff; border: 1px solid var(--line);
    border-left: 4px solid var(--ink);
    border-radius: var(--radius); padding: 14px 16px; margin-bottom: 12px;
}
.panel.tone-green { border-left-color: var(--signal); }
.panel.tone-amber { border-left-color: var(--paint); }
.panel.tone-red   { border-left-color: var(--brake); }
.panel .stat-label { font-size: 0.88rem; color: var(--muted); font-weight: 500; }
.panel .stat-value {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 2rem; font-weight: 600; color: var(--ink); line-height: 1.15; margin-top: 2px;
}
.panel .stat-sub { font-size: 0.82rem; color: var(--muted); margin-top: 2px; }

.step {
    background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
    padding: 14px 16px; height: 100%;
}
.step b { font-family: 'Barlow Semi Condensed', sans-serif; font-size: 1.1rem; }
.step span { display: block; color: var(--muted); font-size: 0.9rem; margin-top: 4px; }
.step i {
    font-style: normal; display: inline-block; width: 22px; height: 22px; line-height: 22px;
    text-align: center; border-radius: 50%; background: var(--paint); color: var(--ink);
    font-weight: 600; font-size: 0.8rem; margin-right: 8px;
}

/* ---------- Native metrics ---------- */
[data-testid="stMetric"] {
    background: #fff; border: 1px solid var(--line); border-radius: var(--radius);
    padding: 12px 14px;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Semi Condensed', sans-serif; font-weight: 600;
}

/* ---------- Badges ---------- */
.badge-green, .badge-yellow, .badge-red, .badge-blue {
    padding: 4px 10px; border-radius: var(--radius); font-size: 0.82rem;
    font-weight: 600; display: inline-block;
}
.badge-green  { background: #DCEFE6; color: #14573F; }
.badge-yellow { background: #FFF0BF; color: #7A5600; }
.badge-red    { background: #F7DAD6; color: #8E2118; }
.badge-blue   { background: #DDE6EC; color: #234559; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: var(--radius); font-weight: 600; border: 1px solid var(--ink);
    background: #fff; color: var(--ink);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--signal); color: var(--signal);
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--signal); border-color: var(--signal); color: #fff;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: #17624A; border-color: #17624A; color: #fff;
}
:focus-visible { outline: 3px solid var(--paint) !important; outline-offset: 2px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--asphalt); border-right: 0; }
[data-testid="stSidebar"] * { color: #E7E9E4; }
[data-testid="stSidebar"] hr { border-color: #343B40; }
.brand { padding: 4px 0 2px 0; }
.brand b {
    font-family: 'Barlow Semi Condensed', sans-serif; font-size: 1.5rem; font-weight: 700;
    color: #fff; display: block; line-height: 1.1;
}
.brand span { font-size: 0.85rem; color: #9AA3A8; }
.brand .lane { height: 3px; margin-top: 12px; }
.chip {
    display: inline-block; padding: 3px 9px; border-radius: var(--radius);
    font-size: 0.78rem; font-weight: 600; background: #2E353A; color: #E7E9E4 !important;
    margin: 0 4px 4px 0;
}
.chip.ok   { background: #1D4E3D; color: #8FE0BE !important; }
.chip.warn { background: #4A3A0A; color: var(--paint) !important; }
.side-label { font-size: 0.8rem; color: #9AA3A8 !important; margin: 14px 0 4px 2px; }

/* Radio -> nav list */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: 9px 12px; border-radius: var(--radius); width: 100%;
    border-left: 3px solid transparent; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 0.98rem; font-weight: 500; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #2A3034; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #2E353A; border-left-color: var(--paint);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #fff; font-weight: 600;
}
[data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--ink) !important; }

@media (max-width: 700px) {
    .page-head h1 { font-size: 1.8rem; }
    .block-container { padding-top: 1.2rem; }
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# UI helpers (presentation only)
# ------------------------------------------------------------------------------
def page_header(title: str, blurb: str) -> None:
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1><p>{blurb}</p><div class='lane'></div></div>",
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def stat_panel(label: str, value: Any, sub: str = "", tone: str = "ink") -> None:
    st.markdown(
        f"""
        <div class='panel tone-{tone}'>
            <div class='stat-label'>{label}</div>
            <div class='stat-value'>{value}</div>
            <div class='stat-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ensure database is initialized
init_db()
dev_info = get_device_info()

# ------------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------------
st.sidebar.markdown(
    "<div class='brand'><b>Smart Traffic Monitor</b>"
    "<span>Vehicle detection, tracking and classification</span>"
    "<div class='lane'></div></div>",
    unsafe_allow_html=True,
)
cuda_class = "ok" if dev_info["cuda_available"] else "warn"
st.sidebar.markdown(
    f"<div style='margin-top:12px'>"
    f"<span class='chip'>PyTorch {dev_info['pytorch_version']}</span>"
    f"<span class='chip {cuda_class}'>{dev_info['device_name']}</span></div>",
    unsafe_allow_html=True,
)

# Display label -> internal page key (keeps the page logic below untouched)
NAV = {
    "Dashboard": "Overview",
    "Dataset Check": "Dataset Explorer",
    "Train": "Train Model",
    "Compare Models": "Model Comparison",
    "Classify Image": "Image Classification",
    "Video Counting": "Live/Video Monitoring",
    "Traffic Insights": "Traffic Analytics",
    "Exports": "Reports & Exports",
    "System & Notes": "Settings & About",
}
st.sidebar.markdown("<div class='side-label'>Go to</div>", unsafe_allow_html=True)
nav_label = st.sidebar.radio("Go to", list(NAV.keys()), label_visibility="collapsed")
page_choice = NAV[nav_label]

# Discover available trained checkpoints
available_checkpoints = sorted(list(MODELS_DIR.glob("*.pth")))
checkpoint_options = {cp.name: cp for cp in available_checkpoints}

st.sidebar.markdown("<div class='side-label'>Classifier in use</div>", unsafe_allow_html=True)
if checkpoint_options:
    selected_cp_name = st.sidebar.selectbox(
        "Classifier in use",
        list(checkpoint_options.keys()),
        index=0,
        label_visibility="collapsed",
    )
    active_checkpoint_path = checkpoint_options[selected_cp_name]
else:
    st.sidebar.caption("No trained model yet. Train one to enable classification.")
    active_checkpoint_path = None

st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0")


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page_choice == "Overview":
    page_header(
        "Dashboard",
        "A snapshot of your hardware, models and recorded traffic runs.",
    )

    experiments = get_experiments()
    sessions = get_traffic_sessions()
    latest_exp = get_latest_experiment()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_panel(
            "Compute device",
            dev_info["current_device"].upper(),
            f"{dev_info['device_name']} ({dev_info.get('total_vram_gb', 0)} GB)",
            tone="green" if dev_info["cuda_available"] else "amber",
        )
    with c2:
        stat_panel("Vehicle classes", len(VEHICLE_CLASSES), "Auto, rickshaw, two-wheelers, bus, truck, car")
    with c3:
        stat_panel("Training runs", len(experiments), "Saved in the local database")
    with c4:
        stat_panel("Video sessions", len(sessions), "Processed traffic videos")

    section("Latest training run")
    if latest_exp:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Model", latest_exp["architecture"])
        e1.caption(f"Status: {latest_exp['status']}")
        e2.metric("Best validation accuracy", f"{latest_exp['best_val_accuracy']*100:.2f}%" if latest_exp['best_val_accuracy'] else "N/A")
        e2.caption(f"Validation loss: {latest_exp['best_val_loss']:.4f}" if latest_exp['best_val_loss'] else "")
        e3.metric("Test accuracy", f"{latest_exp['test_accuracy']*100:.2f}%" if latest_exp['test_accuracy'] else "Not evaluated")
        e3.caption(f"Test F1: {latest_exp['test_macro_f1']*100:.2f}%" if latest_exp['test_macro_f1'] else "")
        e4.metric("Epochs completed", f"{latest_exp['epochs']}")
        e4.caption(f"Took {format_time(latest_exp['training_duration_seconds'] or 0)}")
    else:
        st.info("No training runs yet. Open **Train** to start your first one.")

    section("How the system works")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(
        "<div class='step'><b><i>1</i>Prepare data</b>"
        "<span>Stratified train/val/test split, corrupt-image filtering and inverse-frequency class weights.</span></div>",
        unsafe_allow_html=True,
    )
    s2.markdown(
        "<div class='step'><b><i>2</i>Classify crops</b>"
        "<span>Custom CNN, ResNet-18 and MobileNetV3-Small, compared on the same test split.</span></div>",
        unsafe_allow_html=True,
    )
    s3.markdown(
        "<div class='step'><b><i>3</i>Detect and track</b>"
        "<span>Ultralytics YOLO finds vehicles, ByteTrack gives each one a persistent ID.</span></div>",
        unsafe_allow_html=True,
    )
    s4.markdown(
        "<div class='step'><b><i>4</i>Count and measure</b>"
        "<span>Line-crossing counts without double counting, plus low/medium/high density.</span></div>",
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 2: DATASET CHECK
# ==============================================================================
elif page_choice == "Dataset Explorer":
    page_header(
        "Dataset check",
        "Check folder structure, find corrupt images and see class balance before you train.",
    )

    dataset_path_str = st.text_input("Dataset folder", value=str(DATASET_DIR))
    c_btn1, c_btn2 = st.columns(2)
    run_val = c_btn1.button("Validate dataset", use_container_width=True)
    gen_sample = c_btn2.button("Generate sample dataset (7 classes)", use_container_width=True)

    if gen_sample:
        with st.spinner("Generating sample dataset with 7 vehicle classes..."):
            out_p = create_synthetic_sample_dataset(dataset_path_str, samples_per_class=14)
            st.success(f"Sample dataset created at `{out_p}`. You can validate or train on it now.")

    if run_val or Path(dataset_path_str).exists():
        report = validate_dataset(dataset_path_str)

        if report["is_valid"]:
            st.success(f"Dataset is valid. {report['total_valid_images']} images found.")
        else:
            st.error(f"Dataset has problems: {', '.join(report['errors'])}")

        if report["warnings"]:
            for w in report["warnings"]:
                st.warning(w)

        section("Images per split")
        cols = st.columns(len(report["splits_found"]) if report["splits_found"] else 1)
        for idx, split in enumerate(report["splits_found"]):
            with cols[idx]:
                cnt = sum(report["counts_by_split"].get(split, {}).values())
                st.metric(split.capitalize(), f"{cnt} images")

        if report["counts_by_split"]:
            section("Class distribution")
            df_counts = pd.DataFrame(report["counts_by_split"]).fillna(0).astype(int)
            st.bar_chart(df_counts)

        if report.get("class_imbalance"):
            imb = report["class_imbalance"]
            section("Class balance")
            i1, i2, i3 = st.columns(3)
            i1.metric("Largest / smallest class", f"{imb['imbalance_ratio']}x")
            i2.metric("Severity", imb["severity"])
            i3.metric("Loss function", "Class-weighted" if imb["imbalance_ratio"] > 1.5 else "Standard")


# ==============================================================================
# PAGE 3: TRAIN
# ==============================================================================
elif page_choice == "Train Model":
    page_header(
        "Train a classifier",
        "Train a custom CNN from scratch or fine-tune ResNet-18 or MobileNetV3-Small. "
        "The number of classes is read from your dataset.",
    )

    section("Model and data")
    data_dir_input = st.text_input("Dataset folder", value=str(DATASET_DIR))

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        arch_select = st.selectbox("Model", CLASSIFIER_ARCHITECTURES, index=0)
        epochs_select = st.slider("Epochs", min_value=1, max_value=50, value=5)
        batch_select = st.select_slider("Batch size", options=[4, 8, 16, 32, 64], value=16)
        img_size_select = st.select_slider("Image size (px)", options=[128, 224, 256], value=224)

    with col_cfg2:
        lr_select = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
        weight_decay_select = st.number_input("Weight decay", value=DEFAULT_WEIGHT_DECAY, format="%.5f")
        patience_select = st.number_input("Stop early after (epochs without improvement)", min_value=1, max_value=20, value=DEFAULT_PATIENCE)
        use_weights = st.checkbox("Balance classes with weights", value=True)

    if st.button("Start training", type="primary", use_container_width=True):
        section("Training progress")
        p_bar = st.progress(0)
        status_txt = st.empty()
        metrics_table = st.empty()
        plot_placeholder = st.empty()

        epoch_logs: List[Dict[str, Any]] = []

        def on_epoch_update(curr_epoch: int, total_epochs: int, record: Dict[str, float]):
            p_bar.progress(curr_epoch / total_epochs)
            status_txt.markdown(
                f"**Epoch {curr_epoch}/{total_epochs}** | "
                f"Train loss: `{record['train_loss']}` | Train acc: `{record['train_acc']*100:.1f}%` | "
                f"Val loss: `{record['val_loss']}` | Val acc: `{record['val_acc']*100:.1f}%`"
            )
            epoch_logs.append(record)
            metrics_table.dataframe(pd.DataFrame(epoch_logs).tail(5), use_container_width=True)

        try:
            with st.spinner("Training..."):
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
            st.success(f"Training finished. Checkpoint saved to `{train_res['checkpoint_path']}`")
            if Path(train_res["curves_png"]).exists():
                st.image(train_res["curves_png"], caption="Loss and accuracy curves")
        except Exception as e:
            st.error(f"Training failed: {e}")


# ==============================================================================
# PAGE 4: COMPARE MODELS
# ==============================================================================
elif page_choice == "Model Comparison":
    page_header(
        "Compare models",
        "Run trained classifiers on the same test split and compare accuracy, F1, speed and size.",
    )

    all_cps = sorted(list(MODELS_DIR.glob("*.pth")))
    if not all_cps:
        st.warning("No `.pth` checkpoints found in `models/`. Train a model first.")
    else:
        selected_cps = st.multiselect(
            "Models to compare",
            [c.name for c in all_cps],
            default=[c.name for c in all_cps][:3],
        )
        test_dir_in = st.text_input("Test folder", value=str(DATASET_DIR / "test"))

        if st.button("Run comparison", type="primary") and selected_cps:
            with st.spinner("Benchmarking models..."):
                from compare_models import compare_models
                cp_paths = [MODELS_DIR / name for name in selected_cps]
                try:
                    comp_res = compare_models(
                        checkpoints=cp_paths,
                        test_dir=test_dir_in,
                        output_dir=RESULTS_DIR,
                        device_name="auto",
                    )
                    st.success("Comparison finished.")
                    section("Results")
                    df_comp = pd.DataFrame(comp_res["comparison_table"])
                    st.dataframe(df_comp, use_container_width=True)

                    if Path(comp_res["comparison_plot"]).exists():
                        st.image(comp_res["comparison_plot"], caption="Accuracy, F1 and latency by model")
                except Exception as e:
                    st.error(f"Comparison failed: {e}")


# ==============================================================================
# PAGE 5: CLASSIFY IMAGE
# ==============================================================================
elif page_choice == "Image Classification":
    page_header(
        "Classify an image",
        "Upload one cropped vehicle photo to see the predicted type and how confident the model is.",
    )

    if not active_checkpoint_path:
        st.warning("No trained model found. Train one first, or place a `.pth` file in `models/`.")
    else:
        st.caption(f"Using model: `{active_checkpoint_path.name}`")
        conf_thresh = st.slider("Minimum confidence", 0.1, 1.0, 0.50, 0.05,
                                help="Predictions below this are flagged as uncertain.")

        img_file = st.file_uploader("Vehicle image", type=["jpg", "jpeg", "png", "webp"])

        if img_file is not None:
            pil_image = Image.open(img_file).convert("RGB")
            c_img, c_res = st.columns([1, 1])

            with c_img:
                st.image(pil_image, caption="Your image", use_container_width=True)

            with c_res:
                pred = predict_vehicle_image(
                    image_input=pil_image,
                    checkpoint_path=active_checkpoint_path,
                    confidence_threshold=conf_thresh,
                    device_name="auto",
                )

                if pred["is_uncertain"]:
                    st.markdown(f"<span class='badge-yellow'>{pred['uncertainty_warning']}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='badge-green'>Predicted: {pred['predicted_class'].replace('_', ' ').title()}</span>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                m1.metric("Confidence", f"{pred['confidence_percentage']}%")
                m2.metric("Inference time", f"{pred['inference_latency_ms']} ms", f"on {pred['device'].upper()}")

                st.markdown("**Probability by class**")
                df_p = pd.DataFrame(list(pred["class_probabilities"].items()), columns=["Class", "Probability"])
                st.bar_chart(df_p.set_index("Class"), color=SIGNAL)

            st.caption(pred["educational_notice"])


# ==============================================================================
# PAGE 6: VIDEO COUNTING
# ==============================================================================
elif page_choice == "Live/Video Monitoring":
    page_header(
        "Video counting",
        "Detect and track vehicles in a road video, count line crossings and estimate traffic density.",
    )

    section("Settings")
    v1, v2, v3 = st.columns(3)
    with v1:
        yolo_weight = st.text_input("YOLO weights file", value="yolov8n.pt")
        conf_cut = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
    with v2:
        use_refiner = st.checkbox("Refine vehicle type with classifier", value=bool(active_checkpoint_path))
        refiner_cp = active_checkpoint_path if use_refiner else None
        line_y_ratio = st.slider("Counting line height (% from top)", 20, 80, 55, 5)
    with v3:
        low_max_thresh = st.number_input("Low density: up to (vehicles)", value=4)
        med_max_thresh = st.number_input("Medium density: up to (vehicles)", value=10)

    video_file = st.file_uploader("Road traffic video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.caption(f"{width}x{height} px  |  {total_v_frames} frames  |  {orig_fps:.1f} FPS")

        counting_y = int(height * (line_y_ratio / 100.0))
        counting_line = ((20, counting_y), (width - 20, counting_y))

        start_proc = st.button("Process video", type="primary", use_container_width=True)

        if start_proc:
            # Heavy objects are created only when processing starts
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

            section("Live preview")
            frame_window = st.image([])
            stat_window = st.empty()
            p_bar = st.progress(0)

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

                # 2. Density & counting update
                metrics = density_est.update(tracks, frame.shape, elapsed_seconds=elapsed_sec)

                # 3. Draw annotations
                annotated = tracker.draw_tracks(frame, tracks, draw_trails=True)
                annotated = density_est.draw_density_overlay(annotated, metrics, show_hud=True)

                out_writer.write(annotated)

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

                if frame_idx % 2 == 0 or frame_idx == total_v_frames:
                    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_window.image(rgb_frame, use_container_width=True)
                    if total_v_frames > 0:
                        p_bar.progress(min(1.0, frame_idx / total_v_frames))

                    stat_window.markdown(
                        f"**Frame** `{frame_idx}/{total_v_frames}` | "
                        f"**Density** `{metrics['density_level']}` | "
                        f"**Vehicles counted** `{metrics['total_counted']}` | "
                        f"**Rate** `{metrics['vehicles_per_minute']} per min`"
                    )

            cap.release()
            out_writer.release()

            st.success("Video processed.")

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

            st.info(f"Saved as session #{sess_id}. View it under **Traffic Insights**.")
            st.markdown(f"Annotated video saved to `{out_video_path}`")
        else:
            cap.release()
    else:
        st.info("Upload an MP4, AVI or MOV video to begin.")


# ==============================================================================
# PAGE 7: TRAFFIC INSIGHTS
# ==============================================================================
elif page_choice == "Traffic Analytics":
    page_header(
        "Traffic insights",
        "Vehicle counts, congestion over time and class mix for each processed video.",
    )

    sessions = get_traffic_sessions()
    if not sessions:
        st.info("No sessions yet. Process a video in **Video Counting** first.")
    else:
        session_names = {f"Session #{s['id']} - {s['session_name']} ({s['timestamp'][:10]})": s["id"] for s in sessions}
        picked_name = st.selectbox("Session", list(session_names.keys()))
        picked_id = session_names[picked_name]

        logs = get_traffic_session_logs(picked_id)

        if logs:
            df_logs = pd.DataFrame(logs)

            a1, a2, a3, a4 = st.columns(4)
            s_meta = next(s for s in sessions if s["id"] == picked_id)
            a1.metric("Vehicles counted", s_meta.get("total_vehicles_counted", 0))
            a2.metric("Peak density", s_meta.get("peak_density_level", "Low"))
            a3.metric("Duration", f"{s_meta.get('duration_seconds', 0):.1f}s")
            a4.metric("Average vehicles in view", f"{s_meta.get('avg_density_count', 0):.1f}")

            section("Vehicles in view over time")
            st.line_chart(df_logs.set_index("timestamp_sec")["vehicle_count"], color=SIGNAL)

            section("Vehicle types counted")
            last_counts = s_meta.get("class_counts", {})
            if last_counts:
                df_cls = pd.DataFrame(list(last_counts.items()), columns=["Vehicle Class", "Count"])
                st.bar_chart(df_cls.set_index("Vehicle Class"), color=PAINT)
            else:
                st.caption("No per-class counts were logged for this session.")


# ==============================================================================
# PAGE 8: EXPORTS
# ==============================================================================
elif page_choice == "Reports & Exports":
    page_header(
        "Exports",
        "Download training and traffic session records as CSV.",
    )

    tab1, tab2 = st.tabs(["Training runs", "Video sessions"])

    with tab1:
        experiments = get_experiments()
        if experiments:
            df_exp = pd.DataFrame(experiments)
            st.dataframe(df_exp, use_container_width=True)

            csv_data = df_exp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download training runs (CSV)",
                csv_data,
                "experiments_report.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("No training runs recorded yet.")

    with tab2:
        sessions = get_traffic_sessions()
        if sessions:
            df_sess = pd.DataFrame(sessions)
            st.dataframe(df_sess, use_container_width=True)

            csv_sess = df_sess.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download video sessions (CSV)",
                csv_sess,
                "traffic_sessions_report.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("No video sessions recorded yet.")


# ==============================================================================
# PAGE 9: SYSTEM & NOTES
# ==============================================================================
elif page_choice == "Settings & About":
    page_header(
        "System and notes",
        "Hardware details, how COCO classes map to ours, data sources and known limits.",
    )

    section("Hardware")
    st.json(dev_info)

    section("COCO to 7-class vehicle mapping")
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

    section("Public datasets")
    st.markdown(
        """
        | Dataset | Vehicle classes | License | Notes |
        |---|---|---|---|
        | **Stanford Cars** | Cars (196 sub-models) | Research / CC BY | Cars only; no two-wheelers or auto-rickshaws |
        | **Indian Driving Dataset (IDD)** | Auto-rickshaw, motorcycle, bus, car, truck, bicycle | CC BY-NC 4.0 | Best fit for Indian road conditions |
        | **BDD100K** | Car, bus, truck, motorcycle, bicycle | BSD-3-Clause | Very diverse; no auto-rickshaw |
        | **MIO-TCD** | Motorized and non-motorized classes | Non-commercial | Traffic-camera viewpoints |
        """
    )

    section("Known limits")
    st.warning(DENSITY_DISCLAIMER)
    st.markdown(
        """
        - **Speed estimation:** Real speed (km/h) needs camera calibration to map image pixels onto the road plane. Without calibration markers, any speed value is arbitrary.
        - **Classifier vs. detector:** The classifier only looks at a single cropped vehicle. Finding vehicles in a full road scene needs the YOLO detector.
        """
    )