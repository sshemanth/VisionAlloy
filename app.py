%%writefile app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
import time

st.set_page_config(
    page_title="VisionAlloy | Surface Defect Detection",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    font-size: 18px;
    color: #475569;
}
.metric-card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
}
.success-box {
    background-color: #dcfce7;
    padding: 18px;
    border-radius: 12px;
    color: #166534;
    font-weight: 700;
}
.reject-box {
    background-color: #fee2e2;
    padding: 18px;
    border-radius: 12px;
    color: #991b1b;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 VisionAlloy Surface Defect Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered steel surface inspection using a YOLOv8s model trained in AWS SageMaker</div>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["YOLOv8s - Best Model"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

show_raw_output = st.sidebar.checkbox("Show Raw Prediction Data", value=False)
show_summary = st.sidebar.checkbox("Show Inspection Summary", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Defect Classes")
st.sidebar.write("• Crazing")
st.sidebar.write("• Inclusion")
st.sidebar.write("• Patches")
st.sidebar.write("• Pitted Surface")
st.sidebar.write("• Rolled-in Scale")
st.sidebar.write("• Scratches")

st.sidebar.markdown("---")
st.sidebar.info("Upload a steel surface image and the system will detect visible defects using YOLOv8.")

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

tab1, tab2, tab3 = st.tabs(["🧪 Detection", "📊 Model Performance", "ℹ️ About System"])

with tab1:
    st.subheader("Upload Image for Inspection")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📤 Uploaded Image")
            st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name

        start_time = time.time()

        with st.spinner("Running AI inspection..."):
            results = model.predict(
                source=image_path,
                conf=confidence,
                save=False
            )

        inference_time = time.time() - start_time

        output_image = results[0].plot()
        boxes = results[0].boxes

        with col2:
            st.markdown("### 🎯 Detection Output")
            st.image(output_image, use_container_width=True)

        st.markdown("---")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Detected Defects", len(boxes))

        with m2:
            st.metric("Confidence Threshold", confidence)

        with m3:
            st.metric("Inference Time", f"{inference_time:.2f}s")

        st.markdown("### 🧾 Final Inspection Decision")

        if len(boxes) == 0:
            st.markdown('<div class="success-box">✅ PASS: No defect detected above the selected confidence threshold.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="reject-box">❌ REJECT: Surface defect detected.</div>', unsafe_allow_html=True)

            detection_rows = []

            for box in boxes:
                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])
                class_name = model.names[class_id]

                detection_rows.append({
                    "Defect Class": class_name,
                    "Confidence": round(confidence_score, 3),
                    "Confidence (%)": round(confidence_score * 100, 2)
                })

            detection_table = pd.DataFrame(detection_rows)

            st.markdown("### 📋 Detection Details")
            st.dataframe(detection_table, use_container_width=True)

            st.markdown("### 📊 Defect Count by Class")
            defect_count = detection_table["Defect Class"].value_counts().reset_index()
            defect_count.columns = ["Defect Class", "Count"]
            st.bar_chart(defect_count.set_index("Defect Class"))

        if show_summary:
            st.markdown("### 📝 Inspection Summary")

            if len(boxes) == 0:
                st.write(
                    "The uploaded steel surface image was inspected using the YOLOv8s detection model. "
                    "No surface defect was detected above the selected confidence threshold, so the sample was marked as PASS."
                )
            else:
                st.write(
                    "The uploaded steel surface image was inspected using the YOLOv8s detection model. "
                    "One or more surface defects were detected, so the sample was marked as REJECT."
                )

        if show_raw_output:
            st.markdown("### 🔎 Raw Model Output")
            st.write(results[0].boxes)

        os.remove(image_path)

    else:
        st.warning("Please upload an image to begin inspection.")

with tab2:
    st.subheader("Model Performance Summary")

    performance_data = pd.DataFrame({
        "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m"],
        "Train Accuracy": [67.90, 87.40, 78.85],
        "Validation Accuracy": [67.68, 84.58, 77.84],
        "Test Accuracy": [66.26, 87.56, 78.71]
    })

    st.dataframe(performance_data, use_container_width=True)

    st.markdown("### Accuracy Comparison")
    chart_data = performance_data.set_index("Model")
    st.bar_chart(chart_data)

    st.success("YOLOv8s achieved the best overall performance and was selected as the final model.")

with tab3:
    st.subheader("About This System")

    st.write("""
    VisionAlloy is an automated surface defect detection system designed for industrial quality inspection.
    The system uses a YOLOv8 object detection model trained on the NEU Surface Defect Dataset.
    """)

    st.markdown("### System Workflow")
    st.write("""
    1. Upload a steel surface image  
    2. YOLOv8s detects surface defects  
    3. Bounding boxes are generated  
    4. Defect classes and confidence scores are displayed  
    5. Final PASS/REJECT decision is produced  
    """)

    st.markdown("### Best Model")
    st.write("YOLOv8s was selected because it achieved the highest test accuracy of **87.56%**.")
