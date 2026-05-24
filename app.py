import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import time
from datetime import datetime
import io
import numpy as np

st.set_page_config(
    page_title="VisionAlloy | Automated Inspection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best.pt"
IMAGE_SIZE = 640

DEFECT_DESCRIPTIONS = {
    "crazing": "Fine crack-like defect pattern on the metal surface.",
    "inclusion": "Foreign material or impurity embedded in the surface.",
    "patches": "Irregular surface texture or localised patch defect.",
    "pitted_surface": "Small holes or pits caused by surface damage.",
    "rolled-in_scale": "Scale marks pressed into the metal during rolling.",
    "scratches": "Linear surface damage caused by friction or abrasion."
}

MODEL_RESULTS = pd.DataFrame({
    "Model": ["YOLOv8s"],
    "Train Accuracy": [87.40],
    "Validation Accuracy": [84.58],
    "Test Accuracy": [87.56]
})

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #06111f;
    --panel: #091827;
    --panel2: #0d2033;
    --border: rgba(77, 166, 255, 0.18);
    --text: #dbeafe;
    --muted: #8aa4bf;
    --blue: #2492ff;
    --green: #22c55e;
    --red: #ef4444;
    --orange: #f59e0b;
    --purple: #8b5cf6;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top left, #0b2742 0%, #06111f 35%, #020712 100%);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #03101c 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1500px;
}

.vision-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #7dd3fc;
    margin-bottom: 0;
}

.vision-subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: -0.2rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    color: #c7e6ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.card {
    background: linear-gradient(180deg, rgba(13,32,51,0.92), rgba(7,18,31,0.96));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.05rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.28);
}

.metric-card {
    background: #081a2b;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    min-height: 88px;
}

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.metric-value {
    color: #eef7ff;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

.status-pass {
    color: var(--green);
    font-size: 2rem;
    font-weight: 900;
    text-align: right;
}

.status-reject {
    color: var(--red);
    font-size: 2rem;
    font-weight: 900;
    text-align: right;
}

.small-muted {
    color: var(--muted);
    font-size: 0.86rem;
}

.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #0b7cff, #00b7ff);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
}

.stTabs [data-baseweb="tab"] {
    background: #081a2b;
    border-radius: 10px 10px 0 0;
    border: 1px solid var(--border);
    color: #b7d7f7;
    padding: 0.7rem 1rem;
}

.stTabs [aria-selected="true"] {
    background: #0b76d8 !important;
    color: white !important;
}

[data-testid="stMetric"] {
    background: #081a2b;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem;
}

hr {
    border-color: rgba(125, 211, 252, 0.16);
}

img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return YOLO(path)


def severity_level(confidence):
    if confidence >= 0.80:
        return "Critical"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def run_detection(image, model, confidence_threshold, file_name="uploaded_image"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    start_time = time.time()
    results = model.predict(source=image_path, conf=confidence_threshold, imgsz=IMAGE_SIZE, save=False)
    inference_time = time.time() - start_time

    output_image = results[0].plot()
    boxes = results[0].boxes

    rows = []
    confidences = []

    for box in boxes:
        class_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        class_name = model.names[class_id]
        confidences.append(conf_score)

        rows.append({
            "Defect Class": class_name,
            "Confidence": round(conf_score, 3),
            "Confidence (%)": round(conf_score * 100, 2),
            "Severity": severity_level(conf_score),
            "Description": DEFECT_DESCRIPTIONS.get(class_name, "Surface defect detected.")
        })

    table = pd.DataFrame(rows)
    decision = "PASS" if len(rows) == 0 else "REJECT"
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    os.remove(image_path)
    return output_image, table, decision, inference_time, avg_confidence, confidences, results


st.sidebar.markdown('<div class="vision-title">VisionAlloy</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="vision-subtitle">Automated Inspection Dashboard</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "📤 Upload Image", "📷 Live Capture", "📊 Results", "📈 Analytics", "🕒 History", "ℹ️ About"],
    label_visibility="collapsed"
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False)
st.sidebar.markdown("---")
st.sidebar.success("Model system ready")
st.sidebar.write("Model: **YOLOv8s**")
st.sidebar.write("Image Size: **640 × 640**")

if "history" not in st.session_state:
    st.session_state.history = []

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Place best.pt in the same folder as this app.")
    st.stop()

model = load_model(MODEL_PATH)

st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)

if page in ["🏠 Dashboard", "📤 Upload Image"]:
    uploaded_file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
elif page == "📷 Live Capture":
    uploaded_file = st.camera_input("Capture inspection image")
else:
    uploaded_file = None

if page in ["📊 Results", "📈 Analytics", "🕒 History", "ℹ️ About"]:
    if page == "📊 Results":
        st.markdown('<div class="card">Upload an image from the Dashboard or Upload Image page to generate a result.</div>', unsafe_allow_html=True)
    elif page == "📈 Analytics":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Model Performance")
        st.dataframe(MODEL_RESULTS, use_container_width=True)
        st.bar_chart(MODEL_RESULTS.set_index("Model"))
        st.markdown('</div>', unsafe_allow_html=True)
    elif page == "🕒 History":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Inspection History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
            csv_buffer = io.StringIO()
            history_df.to_csv(csv_buffer, index=False)
            st.download_button("Download History CSV", csv_buffer.getvalue(), "inspection_history.csv", "text/csv")
        else:
            st.info("No inspections completed yet.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card">
        <h3>About VisionAlloy</h3>
        <p class="small-muted">
        VisionAlloy is an automated surface defect inspection dashboard using YOLOv8s object detection.
        It supports upload inspection, live capture, defect confidence scoring, PASS/REJECT decisions,
        and downloadable reports.
        </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

if uploaded_file is None:
    left, right = st.columns([2.2, 1.2])
    with left:
        st.markdown('<div class="card">Upload or capture an image to start the inspection.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Inspection Summary</div>
            <br>
            <div class="small-muted">Overall Status</div>
            <div class="status-pass">READY</div>
            <hr>
            <div class="small-muted">Total Defects: 0</div>
            <div class="small-muted">Avg. Confidence: 0.00</div>
            <div class="small-muted">Model: YOLOv8s</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
output_image, table, decision, inference_time, avg_confidence, confidences, results = run_detection(
    image, model, confidence_threshold, getattr(uploaded_file, "name", "camera_image")
)

st.session_state.history.append({
    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Image": getattr(uploaded_file, "name", "camera_image"),
    "Model": "YOLOv8s",
    "Decision": decision,
    "Defects": len(table),
    "Avg Confidence": round(avg_confidence, 3),
    "Inference Time": round(inference_time, 3)
})

left, right = st.columns([2.25, 1.25], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(output_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    status_class = "status-pass" if decision == "PASS" else "status-reject"
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Inspection Summary</div>
            <br>
            <div class="small-muted">Overall Status</div>
            <div class="{status_class}">{decision}</div>
            <hr>
            <div class="small-muted">Total Defects</div>
            <div style="font-size:1.2rem;font-weight:800;">{len(table)}</div>
            <br>
            <div class="small-muted">Avg. Confidence</div>
            <div style="font-size:1.2rem;font-weight:800;">{avg_confidence:.2f}</div>
            <br>
            <div class="small-muted">Model</div>
            <div style="font-size:1rem;font-weight:700;">YOLOv8s</div>
            <br>
            <div class="small-muted">Image Size</div>
            <div style="font-size:1rem;font-weight:700;">640 × 640</div>
            <br>
            <div class="small-muted">Processing Time</div>
            <div style="font-size:1rem;font-weight:700;">{inference_time:.3f} s</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Confidence Distribution</div>', unsafe_allow_html=True)
    if confidences:
        hist_df = pd.DataFrame({"Confidence Score": confidences})
        st.bar_chart(hist_df)
    else:
        st.info("No confidence scores available.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Decision", decision)
with c2:
    metric_card("Total Defects", len(table))
with c3:
    metric_card("Avg. Confidence", f"{avg_confidence:.2f}")
with c4:
    metric_card("Processing Time", f"{inference_time:.3f}s")

if len(table) > 0:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Detection Details")
    st.dataframe(table, use_container_width=True)

    csv_buffer = io.StringIO()
    table.to_csv(csv_buffer, index=False)
    st.download_button("Download Detection Report CSV", csv_buffer.getvalue(), "inspection_report.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

if show_raw_output:
    st.subheader("Raw Model Output")
    st.write(results[0].boxes)
