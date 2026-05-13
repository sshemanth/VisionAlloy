import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import time
from datetime import datetime
import io

st.set_page_config(
    page_title="VisionAlloy | Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    color: #0f172a;
}
.subtitle {
    font-size: 18px;
    color: #475569;
}
.card {
    background-color: #f8fafc;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
}
.pass-box {
    background-color: #dcfce7;
    color: #166534;
    padding: 18px;
    border-radius: 14px;
    font-size: 20px;
    font-weight: 800;
}
.reject-box {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 18px;
    border-radius: 14px;
    font-size: 20px;
    font-weight: 800;
}
.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 VisionAlloy Surface Defect Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered steel surface inspection using YOLOv8 object detection</div>',
    unsafe_allow_html=True
)

MODEL_PATHS = "YOLOv8s.pt"

MODEL_RESULTS = pd.DataFrame({
    "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m"],
    "Train Accuracy": [67.90, 87.40, 78.85],
    "Validation Accuracy": [67.68, 84.58, 77.84],
    "Test Accuracy": [66.26, 87.56, 78.71]
})

DEFECT_DESCRIPTIONS = {
    "crazing": "Fine crack-like defect pattern on the metal surface.",
    "inclusion": "Foreign material or impurity embedded in the surface.",
    "patches": "Irregular surface texture or localised patch defect.",
    "pitted_surface": "Small holes or pits caused by surface damage.",
    "rolled-in_scale": "Scale marks pressed into the metal during rolling.",
    "scratches": "Linear surface damage caused by friction or abrasion."
}

def severity_level(confidence):
    if confidence >= 0.80:
        return "Critical"
    elif confidence >= 0.50:
        return "Medium"
    return "Low"

@st.cache_resource
def load_model(path):
    return YOLO(path)

st.sidebar.title("⚙️ Control Panel")

st.sidebar.success("Model Loaded: YOLOv8s")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    0.90,
    0.25,
    0.05
)

input_mode = st.sidebar.radio(
    "Input Mode",
    ["Single Image", "Batch Images", "Camera Input"]
)

show_raw_output = st.sidebar.checkbox("Show Raw Output", value=False)
show_defect_info = st.sidebar.checkbox("Show Defect Descriptions", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Model system ready")
st.sidebar.write(f"Selected model: **{model_choice}**")
st.sidebar.write("Task: **Object Detection**")
st.sidebar.write("Image size: **640 × 640**")

model = load_model(MODEL_PATH)

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Detection",
    "📦 Batch Inspection",
    "📊 Model Performance",
    "📜 Inspection History",
    "ℹ️ About"
])

def run_detection(image, file_name="uploaded_image"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    start_time = time.time()

    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=False
    )

    inference_time = time.time() - start_time
    output_image = results[0].plot()
    boxes = results[0].boxes

    rows = []

    for box in boxes:
        class_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        class_name = model.names[class_id]

        rows.append({
            "Defect Class": class_name,
            "Confidence": round(conf_score, 3),
            "Confidence (%)": round(conf_score * 100, 2),
            "Severity": severity_level(conf_score),
            "Description": DEFECT_DESCRIPTIONS.get(class_name, "Surface defect detected.")
        })

    table = pd.DataFrame(rows)

    decision = "PASS" if len(rows) == 0 else "REJECT"

    st.session_state.history.append({
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": file_name,
        "Model": "YOLOv8s",
        "Decision": decision,
        "Defects": len(rows),
        "Inference Time": round(inference_time, 3)
    })

    os.remove(image_path)

    return output_image, table, decision, inference_time, results

with tab1:
    st.subheader("Single Image Inspection")

    if input_mode == "Camera Input":
        uploaded_file = st.camera_input("Capture image")
    else:
        uploaded_file = st.file_uploader(
            "Upload a steel surface image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        output_image, table, decision, inference_time, results = run_detection(
            image,
            getattr(uploaded_file, "name", "camera_image")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📤 Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Detection Output")
            st.image(output_image, use_container_width=True)

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Decision", decision)
        c2.metric("Detected Defects", len(table))
        c3.metric("Inference Time", f"{inference_time:.2f}s")
        c4.metric("Confidence Threshold", confidence_threshold)

        if decision == "PASS":
            st.markdown(
                '<div class="pass-box">✅ PASS: No defect detected above the selected threshold.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="reject-box">❌ REJECT: Surface defect detected.</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 📋 Detection Details")
            st.dataframe(table, use_container_width=True)

            st.markdown("### 📊 Defect Count by Class")
            defect_count = table["Defect Class"].value_counts().reset_index()
            defect_count.columns = ["Defect Class", "Count"]
            st.bar_chart(defect_count.set_index("Defect Class"))

            st.markdown("### 🚨 Severity Distribution")
            severity_count = table["Severity"].value_counts().reset_index()
            severity_count.columns = ["Severity", "Count"]
            st.bar_chart(severity_count.set_index("Severity"))

            if show_defect_info:
                st.markdown("### 🧠 Defect Explanation")
                for _, row in table.iterrows():
                    st.write(
                        f"**{row['Defect Class']}** — {row['Description']} "
                        f"Severity: **{row['Severity']}**."
                    )

            csv_buffer = io.StringIO()
            table.to_csv(csv_buffer, index=False)

            st.download_button(
                label="⬇️ Download Detection Report CSV",
                data=csv_buffer.getvalue(),
                file_name="inspection_report.csv",
                mime="text/csv"
            )

        if show_raw_output:
            st.markdown("### 🔎 Raw Model Output")
            st.write(results[0].boxes)

    else:
        st.info("Upload or capture an image to start inspection.")

with tab2:
    st.subheader("Batch Image Inspection")

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if batch_files:
        batch_results = []

        for file in batch_files:
            image = Image.open(file).convert("RGB")
            output_image, table, decision, inference_time, _ = run_detection(image, file.name)

            batch_results.append({
                "Image": file.name,
                "Decision": decision,
                "Defects": len(table),
                "Inference Time": round(inference_time, 3)
            })

        batch_df = pd.DataFrame(batch_results)

        st.markdown("### Batch Inspection Summary")
        st.dataframe(batch_df, use_container_width=True)

        st.markdown("### Decision Count")
        decision_count = batch_df["Decision"].value_counts().reset_index()
        decision_count.columns = ["Decision", "Count"]
        st.bar_chart(decision_count.set_index("Decision"))

        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ Download Batch Report CSV",
            data=csv_buffer.getvalue(),
            file_name="batch_inspection_report.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Model Performance Dashboard")

    st.dataframe(MODEL_RESULTS, use_container_width=True)

    st.markdown("### Accuracy Comparison")
    st.bar_chart(MODEL_RESULTS.set_index("Model"))

    best_row = MODEL_RESULTS.loc[MODEL_RESULTS["Test Accuracy"].idxmax()]

    st.success(
        f"Best model: YOLOv8s with test accuracy of {best_row['Test Accuracy']}%."
    )

with tab4:
    st.subheader("Inspection History")

    if len(st.session_state.history) == 0:
        st.info("No inspections completed yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ Download History CSV",
            data=csv_buffer.getvalue(),
            file_name="inspection_history.csv",
            mime="text/csv"
        )

with tab5:
    st.subheader("About VisionAlloy")

    st.markdown("""
    VisionAlloy is an automated surface defect inspection system developed using
    YOLOv8 object detection. The system detects six steel surface defect classes:
    
    - Crazing
    - Inclusion
    - Patches
    - Pitted Surface
    - Rolled-in Scale
    - Scratches
    
    The GUI supports single-image inspection, camera input, batch inspection,
    model performance comparison, confidence threshold control, defect severity
    analysis, and downloadable inspection reports.
    """)

    st.markdown("### System Workflow")
    st.write("""
    1. User uploads or captures an image  
    2. YOLOv8 model performs defect detection  
    3. Bounding boxes are generated  
    4. Defect class and confidence score are displayed  
    5. PASS/REJECT decision is produced  
    6. Inspection report can be downloaded  
    """)

st.markdown(
    '<div class="footer">VisionAlloy | Automated Surface Defect Detection System</div>',
    unsafe_allow_html=True
)
