import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import os

st.set_page_config(page_title="Surface Defect Detection", layout="wide")

st.title("Surface Defect Detection System")
st.write("Upload a steel surface image to detect surface defects using the YOLOv8s model trained in AWS SageMaker.")

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    0.90,
    0.25,
    0.05
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    results = model.predict(
        source=image_path,
        conf=confidence,
        save=False
    )

    output_image = results[0].plot()
    boxes = results[0].boxes

    with col2:
        st.subheader("Detection Output")
        st.image(output_image, use_container_width=True)

    st.subheader("Inspection Decision")

    if len(boxes) == 0:
        st.success("PASS: No defect detected")
    else:
        st.error("REJECT: Defect detected")

        rows = []
        for box in boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            class_name = model.names[class_id]

            rows.append({
                "Defect Class": class_name,
                "Confidence": round(confidence_score, 3)
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    os.remove(image_path)
