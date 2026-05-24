import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import tempfile
import os
import time
from datetime import datetime
import io
import numpy as np
import cv2
import wave
import math

st.set_page_config(
    page_title="VisionAlloy | Automated Inspection Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Visual theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg-main: #07111f;
    --bg-card: rgba(12, 25, 43, 0.88);
    --bg-card-soft: rgba(15, 35, 58, 0.78);
    --border: rgba(90, 180, 255, 0.22);
    --cyan: #38bdf8;
    --blue: #2563eb;
    --green: #22c55e;
    --orange: #f59e0b;
    --red: #ef4444;
    --text: #e5f2ff;
    --muted: #9fb4c9;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 34%),
        radial-gradient(circle at bottom right, rgba(37, 99, 235, 0.14), transparent 30%),
        linear-gradient(135deg, #050b14 0%, #081522 52%, #050b14 100%);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101d 0%, #081828 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.18);
}

[data-testid="stSidebar"] * {
    color: #dbeafe !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.hero {
    padding: 22px 26px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(14, 34, 56, 0.96), rgba(8, 20, 34, 0.88));
    border: 1px solid rgba(56, 189, 248, 0.26);
    box-shadow: 0 18px 60px rgba(0,0,0,0.35);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1.2px;
    margin: 0;
    color: #e0f2fe;
}

.hero-title span {
    color: #38bdf8;
    text-shadow: 0 0 24px rgba(56,189,248,0.35);
}

.hero-subtitle {
    color: #a9bfd4;
    font-size: 15px;
    margin-top: 7px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.35);
    color: #86efac;
    font-weight: 700;
    font-size: 13px;
}

.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 16px 50px rgba(0,0,0,0.28);
    margin-bottom: 16px;
}

.panel-title {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.6px;
    color: #dff5ff;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.metric-card {
    background: linear-gradient(180deg, rgba(15, 38, 62, 0.96), rgba(10, 25, 42, 0.96));
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 18px;
    padding: 16px;
    min-height: 112px;
}

.metric-label {
    color: #93a9bd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .4px;
    text-transform: uppercase;
}

.metric-value {
    color: #f8fbff;
    font-size: 28px;
    font-weight: 900;
    margin-top: 7px;
}

.metric-foot {
    color: #8aa2b7;
    font-size: 12px;
    margin-top: 4px;
}

.reject-banner {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.96), rgba(239, 68, 68, 0.22));
    border: 1px solid rgba(248, 113, 113, 0.44);
    color: #fee2e2;
    border-radius: 20px;
    padding: 18px 20px;
    font-size: 20px;
    font-weight: 900;
    box-shadow: 0 12px 34px rgba(239,68,68,0.15);
}

.pass-banner {
    background: linear-gradient(135deg, rgba(20, 83, 45, 0.96), rgba(34, 197, 94, 0.20));
    border: 1px solid rgba(74, 222, 128, 0.44);
    color: #dcfce7;
    border-radius: 20px;
    padding: 18px 20px;
    font-size: 20px;
    font-weight: 900;
    box-shadow: 0 12px 34px rgba(34,197,94,0.12);
}

.review-banner {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.96), rgba(245, 158, 11, 0.20));
    border: 1px solid rgba(251, 191, 36, 0.44);
    color: #fef3c7;
    border-radius: 20px;
    padding: 18px 20px;
    font-size: 20px;
    font-weight: 900;
}

.defect-card {
    background: rgba(9, 22, 38, 0.86);
    border: 1px solid rgba(56, 189, 248, 0.18);
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 10px;
}

.priority-card {
    background: rgba(15, 35, 58, 0.78);
    border-left: 4px solid #38bdf8;
    border-radius: 14px;
    padding: 13px 15px;
    margin-bottom: 10px;
}

.badge-critical {color:#fecaca; background:rgba(239,68,68,.18); border:1px solid rgba(239,68,68,.36); padding:5px 10px; border-radius:999px; font-weight:800;}
.badge-medium {color:#fde68a; background:rgba(245,158,11,.16); border:1px solid rgba(245,158,11,.34); padding:5px 10px; border-radius:999px; font-weight:800;}
.badge-low {color:#bae6fd; background:rgba(56,189,248,.14); border:1px solid rgba(56,189,248,.32); padding:5px 10px; border-radius:999px; font-weight:800;}

hr {border-color: rgba(56,189,248,.14);}
.stTabs [data-baseweb="tab-list"] {gap: 8px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {
    background: rgba(15, 35, 58, 0.82);
    border-radius: 14px;
    border: 1px solid rgba(56,189,248,0.16);
    color: #cfe8ff;
    padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(37,99,235,.95), rgba(56,189,248,.55));
    color: white !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(56,189,248,0.16);
    border-radius: 16px;
    overflow: hidden;
}

.footer {
    text-align:center;
    color:#7f98ad;
    font-size:12px;
    padding:20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Project constants
# -----------------------------
MODEL_PATH = "best.pt"
IMAGE_SIZE = "640 × 640"
FINAL_MODEL = "YOLOv8s"
DEFAULT_CONFIDENCE = 0.25

MODEL_RESULTS = pd.DataFrame({
    "Model": ["YOLOv8n", "YOLOv8s", "YOLOv8m"],
    "Precision": [0.548, 0.770, 0.722],
    "Recall": [0.635, 0.866, 0.727],
    "mAP50": [0.663, 0.876, 0.787],
    "mAP50-95": [0.345, 0.559, 0.455],
})
MODEL_RESULTS["F1 Score"] = (
    2 * MODEL_RESULTS["Precision"] * MODEL_RESULTS["Recall"] /
    (MODEL_RESULTS["Precision"] + MODEL_RESULTS["Recall"])
).round(3)

DEFECT_DESCRIPTIONS = {
    "crazing": "Fine crack-like defect pattern on the metallic surface.",
    "inclusion": "Foreign material or impurity embedded within the surface region.",
    "patches": "Irregular surface texture or localized patch defect.",
    "pitted_surface": "Small holes or pits caused by localized surface damage.",
    "rolled-in_scale": "Scale marks pressed into the material during rolling.",
    "scratches": "Linear surface damage caused by abrasion or friction.",
    "pitted surface": "Small holes or pits caused by localized surface damage.",
    "rolled in scale": "Scale marks pressed into the material during rolling.",
}

RECOMMENDATIONS = {
    "crazing": "Review thermal stress and cooling process. Inspect affected batch before release.",
    "inclusion": "Check raw material purity and contamination sources before continuing production.",
    "patches": "Inspect surface finishing stage and verify coating or rolling consistency.",
    "pitted_surface": "Inspect corrosion, surface wear, or material damage. Separate sample for manual QA review.",
    "pitted surface": "Inspect corrosion, surface wear, or material damage. Separate sample for manual QA review.",
    "rolled-in_scale": "Check rolling process, descaling stage, and roller cleanliness.",
    "rolled in scale": "Check rolling process, descaling stage, and roller cleanliness.",
    "scratches": "Surface polishing or rework is recommended. Check handling and conveyor contact points.",
}

DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "pitted surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "rolled in scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

# -----------------------------
# Utility functions
# -----------------------------
def fmt_name(name: str) -> str:
    key = str(name).strip().lower()
    return DISPLAY_NAMES.get(key, key.replace("_", " ").replace("-", " ").title())


def severity_level(confidence: float) -> str:
    if confidence >= 0.80:
        return "Critical"
    if confidence >= 0.50:
        return "Medium"
    return "Low"


def decision_from_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "PASS"
    critical_count = (table["Severity"] == "Critical").sum()
    avg_conf = table["Confidence"].mean()
    if critical_count > 0 or avg_conf >= 0.70 or len(table) >= 2:
        return "REJECT"
    return "REVIEW"


def generate_inspection_id() -> str:
    if "inspection_counter" not in st.session_state:
        st.session_state.inspection_counter = 0
    st.session_state.inspection_counter += 1
    date_tag = datetime.now().strftime("%Y%m%d")
    return f"VA-{date_tag}-{st.session_state.inspection_counter:03d}"


def calculate_quality_score(table: pd.DataFrame) -> int:
    if table.empty:
        return 100

    penalty = 0
    for _, row in table.iterrows():
        confidence = float(row["Confidence"])
        severity = row["Severity"]
        if severity == "Critical":
            penalty += 35 * confidence
        elif severity == "Medium":
            penalty += 22 * confidence
        else:
            penalty += 12 * confidence

    if len(table) > 1:
        penalty += min(20, (len(table) - 1) * 6)

    score = int(round(100 - penalty))
    return max(0, min(100, score))


def risk_level_from_score(score: int) -> str:
    if score >= 85:
        return "Low"
    if score >= 65:
        return "Medium"
    return "High"


def priority_rank(severity: str, confidence: float) -> int:
    severity_weight = {"Critical": 3, "Medium": 2, "Low": 1}.get(severity, 1)
    return int(severity_weight * 100 + confidence * 100)


def chart_html_bar(label, value, max_value=1.0):
    pct = max(0, min(100, (value / max_value) * 100))
    return f"""
    <div style='margin:10px 0 13px 0;'>
        <div style='display:flex;justify-content:space-between;color:#cfe8ff;font-size:13px;font-weight:700;'>
            <span>{label}</span><span>{value:.3f}</span>
        </div>
        <div style='height:10px;background:rgba(148,163,184,.18);border-radius:999px;margin-top:6px;overflow:hidden;'>
            <div style='height:10px;width:{pct:.1f}%;background:linear-gradient(90deg,#2563eb,#38bdf8);border-radius:999px;'></div>
        </div>
    </div>
    """


def make_confidence_histogram(conf_values):
    if len(conf_values) == 0:
        return pd.DataFrame({
            "Confidence Range": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            "Count": [0, 0, 0, 0, 0]
        })
    bins = [0, .2, .4, .6, .8, 1.0]
    counts, _ = np.histogram(conf_values, bins=bins)
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    return pd.DataFrame({"Confidence Range": labels, "Count": counts})


@st.cache_resource
def load_model(path):
    return YOLO(path)


def create_pdf_report(inspection_id, file_name, decision, quality_score, risk_level, inference_time, table, output_image):
    """Create a simple PDF using PIL only, so no extra package is required."""
    page_w, page_h = 1240, 1754
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 44)
        heading_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        normal_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        title_font = heading_font = normal_font = small_font = ImageFont.load_default()

    y = 55
    draw.text((60, y), "VisionAlloy Inspection Report", fill=(7, 17, 31), font=title_font)
    y += 70
    draw.text((60, y), f"Inspection ID: {inspection_id}", fill=(20, 40, 60), font=normal_font)
    y += 36
    draw.text((60, y), f"Image: {file_name}", fill=(20, 40, 60), font=normal_font)
    y += 36
    draw.text((60, y), f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fill=(20, 40, 60), font=normal_font)
    y += 36
    draw.text((60, y), f"Model: {FINAL_MODEL} | Image Size: {IMAGE_SIZE}", fill=(20, 40, 60), font=normal_font)

    y += 70
    decision_color = (22, 163, 74) if decision == "PASS" else (245, 158, 11) if decision == "REVIEW" else (220, 38, 38)
    draw.rounded_rectangle((60, y, 1180, y + 115), radius=24, outline=decision_color, width=4, fill=(245, 248, 252))
    draw.text((90, y + 25), f"Decision: {decision}", fill=decision_color, font=heading_font)
    draw.text((500, y + 25), f"Quality Score: {quality_score}/100", fill=(7, 17, 31), font=heading_font)
    draw.text((890, y + 25), f"Risk: {risk_level}", fill=(7, 17, 31), font=heading_font)
    y += 155

    draw.text((60, y), "Detection Output", fill=(7, 17, 31), font=heading_font)
    y += 45
    out_pil = Image.fromarray(output_image).convert("RGB") if isinstance(output_image, np.ndarray) else output_image.convert("RGB")
    out_pil.thumbnail((720, 520))
    page.paste(out_pil, (60, y))

    x_table = 820
    draw.text((x_table, y), "Summary", fill=(7, 17, 31), font=heading_font)
    y_summary = y + 50
    draw.text((x_table, y_summary), f"Defects: {len(table)}", fill=(20, 40, 60), font=normal_font)
    y_summary += 34
    draw.text((x_table, y_summary), f"Inference: {inference_time:.3f}s", fill=(20, 40, 60), font=normal_font)
    y_summary += 34
    avg_conf = table["Confidence"].mean() if not table.empty else 0
    draw.text((x_table, y_summary), f"Avg Confidence: {avg_conf:.3f}", fill=(20, 40, 60), font=normal_font)

    y += 560
    draw.text((60, y), "Detected Defects and Recommended Actions", fill=(7, 17, 31), font=heading_font)
    y += 45

    if table.empty:
        draw.text((60, y), "No defects detected above the selected confidence threshold.", fill=(20, 40, 60), font=normal_font)
    else:
        for _, row in table.head(8).iterrows():
            text = f"{row['#']}. {row['Defect Class']} | {row['Severity']} | Conf: {row['Confidence']:.3f}"
            draw.text((60, y), text, fill=(20, 40, 60), font=normal_font)
            y += 30
            rec = str(row.get("Recommendation", "Manual QA review recommended."))
            draw.text((90, y), rec[:95], fill=(70, 85, 100), font=small_font)
            y += 45

    pdf_buffer = io.BytesIO()
    page.save(pdf_buffer, format="PDF", resolution=100.0)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()




def create_region_heatmap(image: Image.Image, table: pd.DataFrame) -> Image.Image:
    """Create a practical explainability heatmap from detected bounding-box regions."""
    base = image.convert("RGB").resize(image.size)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if table.empty:
        return base

    for _, row in table.iterrows():
        try:
            x1, y1, x2, y2 = int(row["X1"]), int(row["Y1"]), int(row["X2"]), int(row["Y2"])
            conf = float(row["Confidence"])
        except Exception:
            continue

        alpha = int(70 + min(150, conf * 150))
        draw.rectangle((x1, y1, x2, y2), fill=(239, 68, 68, alpha), outline=(255, 255, 255, 220), width=3)

        # Draw a soft outer region to visually show model focus around the predicted area.
        pad_x = max(8, int((x2 - x1) * 0.18))
        pad_y = max(8, int((y2 - y1) * 0.18))
        draw.rectangle(
            (max(0, x1 - pad_x), max(0, y1 - pad_y), min(base.width, x2 + pad_x), min(base.height, y2 + pad_y)),
            outline=(56, 189, 248, 160),
            width=2
        )

    blended = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return blended


def create_defect_location_map(table: pd.DataFrame, width: int = 640, height: int = 640) -> Image.Image:
    """Create a normalized factory defect-location heatmap."""
    canvas = Image.new("RGB", (width, height), (7, 17, 31))
    draw = ImageDraw.Draw(canvas, "RGBA")

    # grid
    for x in range(0, width, 80):
        draw.line((x, 0, x, height), fill=(90, 180, 255, 45), width=1)
    for y in range(0, height, 80):
        draw.line((0, y, width, y), fill=(90, 180, 255, 45), width=1)

    if table.empty:
        draw.text((30, 30), "No defect regions detected", fill=(220, 242, 254))
        return canvas

    for _, row in table.iterrows():
        try:
            cx = int((int(row["X1"]) + int(row["X2"])) / 2)
            cy = int((int(row["Y1"]) + int(row["Y2"])) / 2)
            conf = float(row["Confidence"])
        except Exception:
            continue
        radius = int(25 + conf * 45)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(239, 68, 68, 90), outline=(248, 113, 113, 180), width=3)
        draw.ellipse((cx-8, cy-8, cx+8, cy+8), fill=(255, 255, 255, 220))

    draw.text((24, 24), "Defect Location Heatmap", fill=(224, 242, 254))
    return canvas


def generate_predictive_maintenance_insight(history_df: pd.DataFrame) -> str:
    if history_df.empty or "Primary Defect" not in history_df.columns:
        return "No maintenance pattern is available yet. Run more inspections to build a trend."

    valid = history_df[history_df["Primary Defect"] != "None"]
    if valid.empty:
        return "Current inspection history does not show repeated defect patterns. Machine condition appears stable from available samples."

    top_defect = valid["Primary Defect"].value_counts().idxmax()
    top_count = int(valid["Primary Defect"].value_counts().max())
    reject_rate = history_df["Decision"].eq("REJECT").mean() * 100 if "Decision" in history_df.columns else 0

    mapping = {
        "Scratches": "Repeated scratches may indicate conveyor contact, rough handling, or surface friction during transfer.",
        "Rolled-in Scale": "Repeated rolled-in scale may indicate descaling issues, roller contamination, or rolling-stage instability.",
        "Pitted Surface": "Repeated pitted surface defects may indicate corrosion, local surface damage, or material wear before inspection.",
        "Inclusion": "Repeated inclusions may indicate raw material contamination or impurity control problems.",
        "Crazing": "Repeated crazing may indicate thermal stress, cooling inconsistency, or surface cracking behaviour.",
        "Patches": "Repeated patches may indicate surface finishing inconsistency or coating/texture variation."
    }
    cause = mapping.get(top_defect, "Repeated defects suggest a process drift that should be checked by QA engineers.")

    if top_count >= 3 or reject_rate >= 40:
        priority = "High maintenance priority"
    elif top_count == 2 or reject_rate >= 20:
        priority = "Moderate maintenance priority"
    else:
        priority = "Low maintenance priority"

    return f"{priority}: {top_defect} appeared most often ({top_count} time/s). {cause} Current reject rate is {reject_rate:.1f}%."


def make_alert_message(row: dict) -> str:
    return f"""Subject: VisionAlloy Alert - {row.get('Decision', 'Inspection')} for {row.get('Inspection ID', 'Unknown')}

Inspection ID: {row.get('Inspection ID', 'Unknown')}
Time: {row.get('Time', 'Unknown')}
Decision: {row.get('Decision', 'Unknown')}
Risk Level: {row.get('Risk Level', 'Unknown')}
Quality Score: {row.get('Quality Score', 'Unknown')}/100
Primary Defect: {row.get('Primary Defect', 'None')}
Defects Detected: {row.get('Defects', 0)}
Inference Time: {row.get('Inference Time', 0)}s

Recommended action: Please review the inspection output and separate the affected product if the decision is REJECT or the risk level is High.
"""


def create_voice_beep() -> bytes:
    """Generate a short WAV alert tone without external dependencies."""
    sample_rate = 22050
    duration = 0.35
    freq = 880
    n_samples = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * 0.35 * math.sin(2 * math.pi * freq * i / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder='little', signed=True))
    buffer.seek(0)
    return buffer.read()


def predict_frame(frame_bgr, threshold):
    """Run YOLO prediction on a video frame and return summary only."""
    results = model.predict(source=frame_bgr, conf=threshold, save=False, verbose=False)
    plotted = results[0].plot()
    boxes = results[0].boxes
    count = len(boxes)
    avg_conf = 0.0
    primary = "None"
    if count > 0:
        confs = [float(b.conf[0]) for b in boxes]
        avg_conf = float(np.mean(confs))
        best = boxes[int(np.argmax(confs))]
        primary = fmt_name(model.names[int(best.cls[0])])
    decision = "PASS" if count == 0 else "REJECT" if count >= 2 or avg_conf >= 0.70 else "REVIEW"
    return plotted, count, avg_conf, primary, decision

def run_detection(image, file_name, threshold):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    inspection_id = generate_inspection_id()
    start_time = time.time()
    results = model.predict(source=image_path, conf=threshold, save=False, verbose=False)
    inference_time = time.time() - start_time

    output_image = results[0].plot()
    boxes = results[0].boxes
    rows = []
    crops = []

    image_np = np.array(image)
    width, height = image.size

    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        raw_class = str(model.names[class_id])
        class_name = fmt_name(raw_class)
        raw_key = raw_class.strip().lower()
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        sev = severity_level(conf_score)
        recommendation = RECOMMENDATIONS.get(raw_key, "Manual inspection and QA review recommended.")
        rows.append({
            "#": i + 1,
            "Defect Class": class_name,
            "Confidence": round(conf_score, 3),
            "Confidence (%)": round(conf_score * 100, 2),
            "Severity": sev,
            "Priority": priority_rank(sev, conf_score),
            "Box": f"({x1}, {y1}) - ({x2}, {y2})",
            "X1": x1,
            "Y1": y1,
            "X2": x2,
            "Y2": y2,
            "Description": DEFECT_DESCRIPTIONS.get(raw_key, "Surface defect detected."),
            "Recommendation": recommendation
        })

        if x2 > x1 and y2 > y1:
            crops.append({
                "image": image_np[y1:y2, x1:x2],
                "class": class_name,
                "confidence": conf_score,
                "severity": sev
            })

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(by="Priority", ascending=False).reset_index(drop=True)
        table["#"] = range(1, len(table) + 1)

    decision = decision_from_table(table)
    quality_score = calculate_quality_score(table)
    risk_level = risk_level_from_score(quality_score)
    primary_defect = table.iloc[0]["Defect Class"] if not table.empty else "None"

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Inspection ID": inspection_id,
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image": file_name,
        "Model": FINAL_MODEL,
        "Decision": decision,
        "Risk Level": risk_level,
        "Quality Score": quality_score,
        "Primary Defect": primary_defect,
        "Defects": len(table),
        "Avg Confidence": round(table["Confidence"].mean(), 3) if not table.empty else 0,
        "Inference Time": round(inference_time, 3)
    })

    try:
        os.remove(image_path)
    except OSError:
        pass

    return output_image, table, decision, inference_time, results, crops, inspection_id, quality_score, risk_level


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("""
<div style='padding:12px 4px 18px 4px;'>
    <div style='font-size:30px;font-weight:900;color:#7dd3fc;'>VisionAlloy</div>
    <div style='font-size:12px;color:#9fb4c9;margin-top:2px;'>Automated Inspection Dashboard</div>
</div>
""", unsafe_allow_html=True)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_CONFIDENCE,
    step=0.05
)

input_mode = st.sidebar.radio(
    "Inspection Mode",
    ["Single Image", "Batch Images", "Camera Input"],
    index=0
)

show_defect_info = st.sidebar.checkbox("Show defect explanations", value=True)
show_recommendations = st.sidebar.checkbox("Show smart recommendations", value=True)
show_raw_output = st.sidebar.checkbox("Show raw YOLO output", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success("Model system ready")
st.sidebar.write(f"Model: **{FINAL_MODEL}**")
st.sidebar.write("Task: **Object Detection**")
st.sidebar.write(f"Input size: **{IMAGE_SIZE}**")
st.sidebar.write("Dataset: **NEU Surface Defect Dataset**")

# -----------------------------
# Load model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Keep best.pt in the same folder as this app.py file.")
    st.stop()

model = load_model(MODEL_PATH)

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class='hero'>
    <div style='display:flex;justify-content:space-between;gap:20px;align-items:center;flex-wrap:wrap;'>
        <div>
            <h1 class='hero-title'><span>Vision</span>Alloy</h1>
            <div class='hero-subtitle'>Vision-based automated metallic surface inspection using YOLOv8s, real-time defect localization, confidence analysis, quality scoring, and PASS / REVIEW / REJECT decision support.</div>
        </div>
        <div class='status-pill'>● LIVE INSPECTION READY</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🏭 Inspection Dashboard",
    "🎥 Video Inspection",
    "📦 Batch Inspection",
    "📊 Model Performance",
    "🏢 Factory Analytics",
    "🔥 Explainability",
    "🕒 Inspection History",
    "🧠 Production Intelligence",
    "ℹ️ System Details"
])

with tab1:
    left, right = st.columns([1.9, 1], gap="large")

    with left:
        st.markdown("<div class='panel'><div class='panel-title'>Detection Result</div>", unsafe_allow_html=True)

        if input_mode == "Camera Input":
            uploaded_file = st.camera_input("Capture surface image")
        else:
            uploaded_file = st.file_uploader(
                "Upload a metallic surface image",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                label_visibility="collapsed"
            )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner("Running YOLOv8s surface inspection..."):
                output_image, table, decision, inference_time, results, crops, inspection_id, quality_score, risk_level = run_detection(
                    image,
                    getattr(uploaded_file, "name", "camera_image"),
                    confidence_threshold
                )

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.caption("Input Image")
                st.image(image, use_container_width=True)
            with img_col2:
                st.caption("Detected Output")
                st.image(output_image, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            defects = len(table)
            avg_conf = table["Confidence"].mean() if not table.empty else 0
            max_conf = table["Confidence"].max() if not table.empty else 0

            st.markdown(f"**Inspection ID:** `{inspection_id}`")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Overall Status</div><div class='metric-value'>{decision}</div><div class='metric-foot'>Automated QA decision</div></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Quality Score</div><div class='metric-value'>{quality_score}/100</div><div class='metric-foot'>Surface health estimate</div></div>", unsafe_allow_html=True)
            with m3:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value'>{risk_level}</div><div class='metric-foot'>Based on severity and confidence</div></div>", unsafe_allow_html=True)
            with m4:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Defects</div><div class='metric-value'>{defects}</div><div class='metric-foot'>Above threshold</div></div>", unsafe_allow_html=True)
            with m5:
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Processing Time</div><div class='metric-value'>{inference_time:.3f}s</div><div class='metric-foot'>Single image inference</div></div>", unsafe_allow_html=True)

            if decision == "PASS":
                st.markdown("<div class='pass-banner'>PASS — No defect detected above the selected confidence threshold.</div>", unsafe_allow_html=True)
            elif decision == "REVIEW":
                st.markdown("<div class='review-banner'>REVIEW — Low or moderate confidence defect detected. Manual verification is recommended.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='reject-banner'>REJECT — Surface defect detected. Product should be separated for quality review.</div>", unsafe_allow_html=True)

            if not table.empty:
                st.markdown("### Defect Details")
                st.dataframe(table, use_container_width=True, hide_index=True)

                if len(crops) > 0:
                    st.markdown("### Defect Crop Preview")
                    crop_cols = st.columns(min(4, len(crops)))
                    for idx, crop in enumerate(crops[:8]):
                        with crop_cols[idx % len(crop_cols)]:
                            st.image(
                                crop["image"],
                                caption=f"{crop['class']} | {crop['confidence']:.2f} | {crop['severity']}",
                                use_container_width=True
                            )

                if show_defect_info:
                    st.markdown("### Defect Interpretation")
                    for _, row in table.iterrows():
                        badge_class = "badge-critical" if row["Severity"] == "Critical" else "badge-medium" if row["Severity"] == "Medium" else "badge-low"
                        st.markdown(
                            f"<div class='defect-card'><b>{row['Defect Class']}</b> &nbsp; <span class='{badge_class}'>{row['Severity']}</span><br><span style='color:#9fb4c9;'>{row['Description']}</span></div>",
                            unsafe_allow_html=True
                        )

                if show_recommendations:
                    st.markdown("### Smart Industrial Recommendations")
                    rec_table = table[["#", "Defect Class", "Severity", "Confidence", "Recommendation"]].copy()
                    st.dataframe(rec_table, use_container_width=True, hide_index=True)

                    for _, row in rec_table.iterrows():
                        st.markdown(
                            f"<div class='priority-card'><b>Priority {row['#']} — {row['Defect Class']}</b><br><span style='color:#9fb4c9;'>Severity: {row['Severity']} | Confidence: {row['Confidence']:.3f}</span><br>{row['Recommendation']}</div>",
                            unsafe_allow_html=True
                        )

                csv_buffer = io.StringIO()
                table.to_csv(csv_buffer, index=False)

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "⬇️ Download Inspection CSV",
                        csv_buffer.getvalue(),
                        file_name=f"{inspection_id}_inspection_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d2:
                    pdf_bytes = create_pdf_report(
                        inspection_id=inspection_id,
                        file_name=getattr(uploaded_file, "name", "camera_image"),
                        decision=decision,
                        quality_score=quality_score,
                        risk_level=risk_level,
                        inference_time=inference_time,
                        table=table,
                        output_image=output_image
                    )
                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf_bytes,
                        file_name=f"{inspection_id}_inspection_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                pdf_bytes = create_pdf_report(
                    inspection_id=inspection_id,
                    file_name=getattr(uploaded_file, "name", "camera_image"),
                    decision=decision,
                    quality_score=quality_score,
                    risk_level=risk_level,
                    inference_time=inference_time,
                    table=table,
                    output_image=output_image
                )
                st.download_button(
                    "⬇️ Download PDF Report",
                    pdf_bytes,
                    file_name=f"{inspection_id}_inspection_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            if show_raw_output:
                st.markdown("### Raw YOLO Output")
                st.write(results[0].boxes)

        else:
            st.info("Upload or capture an image to start inspection.")
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'><div class='panel-title'>Inspection Summary</div>", unsafe_allow_html=True)
        if uploaded_file is None:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-label'>Overall Status</div>
                <div class='metric-value'>WAITING</div>
                <div class='metric-foot'>No image inspected yet</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(chart_html_bar("YOLOv8s mAP50", 0.876), unsafe_allow_html=True)
            st.markdown(chart_html_bar("YOLOv8s Recall", 0.866), unsafe_allow_html=True)
            st.markdown(chart_html_bar("YOLOv8s Precision", 0.770), unsafe_allow_html=True)
        else:
            status_color = "#22c55e" if decision == "PASS" else "#f59e0b" if decision == "REVIEW" else "#ef4444"
            st.markdown(f"""
            <div style='font-size:13px;color:#9fb4c9;font-weight:700;'>Overall Status</div>
            <div style='font-size:36px;color:{status_color};font-weight:900;margin-bottom:10px;'>{decision}</div>
            <hr>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
                <div><div style='color:#8aa2b7;font-size:12px;'>Inspection ID</div><div style='font-size:16px;font-weight:900;'>{inspection_id}</div></div>
                <div><div style='color:#8aa2b7;font-size:12px;'>Risk</div><div style='font-size:22px;font-weight:900;'>{risk_level}</div></div>
                <div><div style='color:#8aa2b7;font-size:12px;'>Quality</div><div style='font-size:22px;font-weight:900;'>{quality_score}/100</div></div>
                <div><div style='color:#8aa2b7;font-size:12px;'>Defects</div><div style='font-size:22px;font-weight:900;'>{defects}</div></div>
                <div><div style='color:#8aa2b7;font-size:12px;'>Model</div><div style='font-size:22px;font-weight:900;'>{FINAL_MODEL}</div></div>
                <div><div style='color:#8aa2b7;font-size:12px;'>Max Conf.</div><div style='font-size:22px;font-weight:900;'>{max_conf:.2f}</div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'><div class='panel-title'>Confidence Distribution</div>", unsafe_allow_html=True)
        if uploaded_file is not None and not table.empty:
            hist_df = make_confidence_histogram(table["Confidence"].tolist())
            st.bar_chart(hist_df.set_index("Confidence Range"), use_container_width=True)
        else:
            placeholder = pd.DataFrame({
                "Confidence Range": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
                "Count": [0, 0, 0, 0, 0]
            })
            st.bar_chart(placeholder.set_index("Confidence Range"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='panel'><div class='panel-title'>Supported Defect Classes</div>", unsafe_allow_html=True)
        st.markdown("Crazing · Inclusion · Patches · Pitted Surface · Rolled-in Scale · Scratches")
        st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown("<div class='panel'><div class='panel-title'>Real-Time Video Inspection</div>", unsafe_allow_html=True)
    st.write("Upload an MP4/AVI/MOV file. The app will process frames and display live YOLO bounding boxes during inspection.")

    video_file = st.file_uploader(
        "Upload inspection video",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=False,
        key="video_inspection_upload"
    )

    frame_step = st.slider("Process every Nth frame", 1, 60, 5, 1)
    max_frames = st.slider("Maximum processed frames", 5, 150, 50, 5)
    video_confidence = st.slider("Video Confidence Threshold", 0.05, 0.90, 0.10, 0.05)

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0

        st.write(f"Video frames: **{total_frames}** | FPS: **{fps:.2f}**")
        st.info("For live defect display, the app shows the processed YOLO frames below instead of the raw uploaded video.")

        run_video = st.button("Run Live Video Inspection", use_container_width=True)

        live_frame_box = st.empty()
        live_status_box = st.empty()
        progress = st.progress(0)

        if run_video:
            summaries = []
            preview_images = []
            processed = 0
            frame_idx = 0
            start_video_time = time.time()

            while cap.isOpened() and processed < max_frames:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx % frame_step == 0:
                    # Resize every sampled frame to the same scale used during YOLO training.
                    frame_resized = cv2.resize(frame, (640, 640))

                    # Run YOLO prediction with a lower threshold for video because motion blur can reduce confidence.
                    results = model.predict(
                        source=frame_resized,
                        conf=video_confidence,
                        save=False,
                        verbose=False
                    )

                    # This is the important line: display the YOLO annotated frame, not the original frame.
                    annotated_frame = results[0].plot()
                    boxes = results[0].boxes
                    defect_count = len(boxes)

                    if defect_count > 0:
                        confs = [float(b.conf[0]) for b in boxes]
                        avg_conf = float(np.mean(confs))
                        best_box = boxes[int(np.argmax(confs))]
                        primary_defect = fmt_name(model.names[int(best_box.cls[0])])
                    else:
                        avg_conf = 0.0
                        primary_defect = "None"

                    frame_decision = "PASS" if defect_count == 0 else "REJECT" if defect_count >= 2 or avg_conf >= 0.70 else "REVIEW"

                    summaries.append({
                        "Frame": frame_idx,
                        "Decision": frame_decision,
                        "Defects": defect_count,
                        "Avg Confidence": round(avg_conf, 3),
                        "Primary Defect": primary_defect
                    })

                    # Display live annotated frame in RGB format for Streamlit.
                    live_frame_box.image(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Live YOLO Detection | Frame {frame_idx} | {frame_decision} | Defects: {defect_count}",
                        use_container_width=True
                    )

                    live_status_box.markdown(
                        f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Live Video Status</div>
                            <div class='metric-value'>{frame_decision}</div>
                            <div class='metric-foot'>Frame {frame_idx} | Defects: {defect_count} | Avg confidence: {avg_conf:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if len(preview_images) < 8:
                        preview_images.append(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

                    processed += 1
                    progress.progress(min(1.0, processed / max_frames))

                frame_idx += 1

            cap.release()

            try:
                os.remove(video_path)
            except OSError:
                pass

            if summaries:
                video_df = pd.DataFrame(summaries)
                reject_rate = video_df["Decision"].eq("REJECT").mean() * 100
                review_rate = video_df["Decision"].eq("REVIEW").mean() * 100
                avg_defects = video_df["Defects"].mean()
                runtime = time.time() - start_video_time

                v1, v2, v3, v4 = st.columns(4)
                v1.metric("Processed Frames", len(video_df))
                v2.metric("Reject Rate", f"{reject_rate:.1f}%")
                v3.metric("Review Rate", f"{review_rate:.1f}%")
                v4.metric("Runtime", f"{runtime:.2f}s")

                st.dataframe(video_df, use_container_width=True, hide_index=True)

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("### Frame Decision Distribution")
                    decision_counts = video_df["Decision"].value_counts().reset_index()
                    decision_counts.columns = ["Decision", "Count"]
                    st.bar_chart(decision_counts.set_index("Decision"), use_container_width=True)

                with c2:
                    st.markdown("### Defects Across Processed Frames")
                    st.line_chart(video_df.set_index("Frame")[["Defects"]], use_container_width=True)

                st.markdown("### Annotated Frame Samples")
                cols = st.columns(min(4, len(preview_images)))
                for i, img in enumerate(preview_images):
                    with cols[i % len(cols)]:
                        st.image(img, caption=f"Detected sample {i + 1}", use_container_width=True)

                csv_buffer = io.StringIO()
                video_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download Video Inspection CSV",
                    csv_buffer.getvalue(),
                    file_name="visionalloy_video_inspection.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No frames were processed. Try reducing the frame step or using a shorter video.")

        else:
            cap.release()
            try:
                os.remove(video_path)
            except OSError:
                pass
    else:
        st.info("Upload a video to start frame-based inspection.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='panel'><div class='panel-title'>Batch Inspection</div>", unsafe_allow_html=True)
    batch_files = st.file_uploader(
        "Upload multiple surface images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if batch_files:
        batch_results = []
        progress = st.progress(0)
        for idx, file in enumerate(batch_files):
            image = Image.open(file).convert("RGB")
            _, table, decision, inference_time, _, _, inspection_id, quality_score, risk_level = run_detection(
                image, file.name, confidence_threshold
            )
            batch_results.append({
                "Inspection ID": inspection_id,
                "Image": file.name,
                "Decision": decision,
                "Risk Level": risk_level,
                "Quality Score": quality_score,
                "Defects": len(table),
                "Avg Confidence": round(table["Confidence"].mean(), 3) if not table.empty else 0,
                "Inference Time": round(inference_time, 3)
            })
            progress.progress((idx + 1) / len(batch_files))

        batch_df = pd.DataFrame(batch_results)
        st.dataframe(batch_df, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Decision Count")
            decision_counts = batch_df["Decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]
            st.bar_chart(decision_counts.set_index("Decision"), use_container_width=True)
        with c2:
            st.markdown("### Defects per Image")
            st.bar_chart(batch_df.set_index("Image")[["Defects"]], use_container_width=True)
        with c3:
            st.markdown("### Quality Score")
            st.bar_chart(batch_df.set_index("Image")[["Quality Score"]], use_container_width=True)

        csv_buffer = io.StringIO()
        batch_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Batch Report CSV",
            csv_buffer.getvalue(),
            file_name="visionalloy_batch_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Upload multiple images to run batch inspection.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='panel'><div class='panel-title'>Model Performance Dashboard</div>", unsafe_allow_html=True)
    st.dataframe(MODEL_RESULTS, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### mAP Comparison")
        st.bar_chart(MODEL_RESULTS.set_index("Model")[["mAP50", "mAP50-95"]], use_container_width=True)
    with c2:
        st.markdown("### Detection Reliability")
        st.bar_chart(MODEL_RESULTS.set_index("Model")[["Precision", "Recall", "F1 Score"]], use_container_width=True)

    st.success("Final selected model: YOLOv8s. It gives the strongest balance between localization accuracy and computational efficiency for the NEU surface defect dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown("<div class='panel'><div class='panel-title'>Factory Analytics</div>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("Factory analytics will appear after at least one inspection is completed.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        total_inspections = len(history_df)
        reject_rate = (history_df["Decision"].eq("REJECT").mean() * 100) if "Decision" in history_df.columns else 0
        review_rate = (history_df["Decision"].eq("REVIEW").mean() * 100) if "Decision" in history_df.columns else 0
        avg_quality = history_df["Quality Score"].mean() if "Quality Score" in history_df.columns else 0
        avg_time = history_df["Inference Time"].mean() if "Inference Time" in history_df.columns else 0

        valid_defects = history_df[history_df["Primary Defect"] != "None"] if "Primary Defect" in history_df.columns else pd.DataFrame()
        most_common_defect = valid_defects["Primary Defect"].mode().iloc[0] if not valid_defects.empty else "None"

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Inspections", total_inspections)
        with k2:
            st.metric("Reject Rate", f"{reject_rate:.1f}%")
        with k3:
            st.metric("Review Rate", f"{review_rate:.1f}%")
        with k4:
            st.metric("Avg Quality", f"{avg_quality:.1f}/100")
        with k5:
            st.metric("Avg Time", f"{avg_time:.3f}s")

        st.markdown(f"**Most common defect:** `{most_common_defect}`")

        a1, a2 = st.columns(2)
        with a1:
            st.markdown("### Production Decision Distribution")
            decision_counts = history_df["Decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]
            st.bar_chart(decision_counts.set_index("Decision"), use_container_width=True)
        with a2:
            st.markdown("### Risk Level Distribution")
            if "Risk Level" in history_df.columns:
                risk_counts = history_df["Risk Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                st.bar_chart(risk_counts.set_index("Risk Level"), use_container_width=True)

        b1, b2 = st.columns(2)
        with b1:
            st.markdown("### Quality Score Trend")
            st.line_chart(history_df[["Quality Score"]], use_container_width=True)
        with b2:
            st.markdown("### Defect Frequency")
            if not valid_defects.empty:
                defect_counts = valid_defects["Primary Defect"].value_counts().reset_index()
                defect_counts.columns = ["Defect", "Count"]
                st.bar_chart(defect_counts.set_index("Defect"), use_container_width=True)
            else:
                st.info("No defects recorded yet.")

    st.markdown("</div>", unsafe_allow_html=True)


with tab6:
    st.markdown("<div class='panel'><div class='panel-title'>Explainability and Defect Heatmaps</div>", unsafe_allow_html=True)
    st.write("This module visualizes model attention using detected bounding-box regions. It is a practical inspection heatmap, not a full Grad-CAM implementation.")

    explain_file = st.file_uploader(
        "Upload image for explainability analysis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="explainability_upload"
    )

    if explain_file is not None:
        explain_image = Image.open(explain_file).convert("RGB")
        with st.spinner("Generating detection and explainability heatmap..."):
            output_image, table, decision, inference_time, results, crops, inspection_id, quality_score, risk_level = run_detection(
                explain_image,
                getattr(explain_file, "name", "explainability_image"),
                confidence_threshold
            )
            heatmap_img = create_region_heatmap(explain_image, table)
            location_map = create_defect_location_map(table, explain_image.width, explain_image.height)

        e1, e2, e3 = st.columns(3)
        with e1:
            st.caption("Original")
            st.image(explain_image, use_container_width=True)
        with e2:
            st.caption("YOLO Detection")
            st.image(output_image, use_container_width=True)
        with e3:
            st.caption("Explainability Heatmap")
            st.image(heatmap_img, use_container_width=True)

        h1, h2 = st.columns([1, 1])
        with h1:
            st.markdown("### Defect Location Map")
            st.image(location_map, use_container_width=True)
        with h2:
            st.markdown("### Interpretation")
            st.write(f"Inspection ID: `{inspection_id}`")
            st.write(f"Decision: **{decision}**")
            st.write(f"Quality Score: **{quality_score}/100**")
            st.write(f"Risk Level: **{risk_level}**")
            if table.empty:
                st.success("No defect region was detected above the selected threshold.")
            else:
                st.dataframe(table[["Defect Class", "Confidence", "Severity", "Recommendation"]], use_container_width=True, hide_index=True)

        heatmap_buffer = io.BytesIO()
        heatmap_img.save(heatmap_buffer, format="PNG")
        st.download_button(
            "⬇️ Download Explainability Heatmap",
            heatmap_buffer.getvalue(),
            file_name=f"{inspection_id}_heatmap.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("Upload an image to generate an explainability heatmap.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab7:
    st.markdown("<div class='panel'><div class='panel-title'>Inspection History</div>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No inspections completed yet.")
    else:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Decision Trend")
            if "Decision" in history_df.columns and not history_df.empty:
                decision_counts = history_df["Decision"].value_counts().reset_index()
                decision_counts.columns = ["Decision", "Count"]
                st.bar_chart(decision_counts.set_index("Decision"), use_container_width=True)
            else:
                st.info("No inspection decision data available yet.")

        with col_b:
            st.markdown("### Inference Time")
            if "Inference Time" in history_df.columns:
                st.line_chart(history_df[["Inference Time"]], use_container_width=True)

        csv_buffer = io.StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download History CSV",
            csv_buffer.getvalue(),
            file_name="visionalloy_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


with tab8:
    st.markdown("<div class='panel'><div class='panel-title'>Production Intelligence</div>", unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history) if len(st.session_state.history) > 0 else pd.DataFrame()

    st.markdown("### Conveyor Belt Simulation")
    if history_df.empty:
        st.info("Run inspections first to populate the conveyor simulation.")
        recent_rows = []
    else:
        recent_rows = history_df.tail(6).to_dict("records")

    belt_html = "<div style='background:linear-gradient(90deg,#0f2238,#1e3a5f,#0f2238);border:1px solid rgba(56,189,248,.25);border-radius:22px;padding:18px;display:flex;gap:14px;overflow-x:auto;'>"
    if not recent_rows:
        for i in range(6):
            belt_html += f"<div style='min-width:150px;height:90px;border-radius:16px;background:rgba(148,163,184,.14);border:1px dashed rgba(203,213,225,.25);display:flex;align-items:center;justify-content:center;color:#9fb4c9;font-weight:800;'>Waiting</div>"
    else:
        for row in recent_rows:
            decision = row.get("Decision", "WAIT")
            color = "#22c55e" if decision == "PASS" else "#f59e0b" if decision == "REVIEW" else "#ef4444"
            belt_html += f"""
            <div style='min-width:180px;border-radius:16px;background:rgba(7,17,31,.72);border:1px solid {color};padding:12px;'>
                <div style='width:14px;height:14px;background:{color};border-radius:999px;box-shadow:0 0 16px {color};'></div>
                <div style='font-size:13px;color:#dff5ff;font-weight:900;margin-top:10px;'>{row.get('Inspection ID','VA')}</div>
                <div style='font-size:24px;color:{color};font-weight:900;'>{decision}</div>
                <div style='font-size:12px;color:#9fb4c9;'>Score: {row.get('Quality Score','-')}/100</div>
            </div>
            """
    belt_html += "</div>"
    st.markdown(belt_html, unsafe_allow_html=True)

    st.markdown("### Predictive Maintenance Insight")
    insight = generate_predictive_maintenance_insight(history_df)
    st.markdown(f"<div class='priority-card'>{insight}</div>", unsafe_allow_html=True)

    st.markdown("### Threshold Auto-Optimization Suggestion")
    if history_df.empty:
        st.info("Not enough inspection history to suggest a threshold adjustment.")
    else:
        reject_rate = history_df["Decision"].eq("REJECT").mean() * 100 if "Decision" in history_df.columns else 0
        avg_conf = history_df["Avg Confidence"].mean() if "Avg Confidence" in history_df.columns else 0
        if reject_rate > 55 and avg_conf < 0.45:
            suggestion = "False positives may be high. Consider increasing the confidence threshold by 0.05."
        elif reject_rate < 10 and avg_conf > 0.70:
            suggestion = "The system is conservative. If small defects are being missed, consider lowering the threshold by 0.05."
        else:
            suggestion = "Current threshold appears reasonable based on available inspection history."
        st.write(suggestion)

    st.markdown("### Operator Notes and Manual QA Review")
    if "operator_notes" not in st.session_state:
        st.session_state.operator_notes = {}

    if history_df.empty:
        st.info("No inspection records available for notes.")
    else:
        selected_id = st.selectbox("Select Inspection ID", history_df["Inspection ID"].tolist()[::-1])
        current_note = st.session_state.operator_notes.get(selected_id, "")
        note = st.text_area("Operator / QA note", value=current_note, height=120)
        if st.button("Save QA Note", use_container_width=True):
            st.session_state.operator_notes[selected_id] = note
            st.success("QA note saved for this inspection.")

    st.markdown("### Critical Alert Assistant")
    if history_df.empty:
        st.info("No alert can be generated yet.")
    else:
        latest = history_df.iloc[-1].to_dict()
        alert_text = make_alert_message(latest)
        st.text_area("Supervisor alert template", value=alert_text, height=230)
        st.download_button(
            "⬇️ Download Alert Message",
            alert_text,
            file_name=f"{latest.get('Inspection ID','visionalloy')}_alert.txt",
            mime="text/plain",
            use_container_width=True
        )
        if latest.get("Decision") == "REJECT" or latest.get("Risk Level") == "High":
            st.audio(create_voice_beep(), format="audio/wav")
            st.warning("Critical alert tone is available above for demo use.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab9:
    st.markdown("<div class='panel'><div class='panel-title'>System Details</div>", unsafe_allow_html=True)
    st.markdown("""
    **VisionAlloy** is a vision-based automated inspection system for metallic surface defect detection.  
    The system uses a trained **YOLOv8s** object detection model to localize defects with bounding boxes, confidence scores, severity levels, inspection decisions, quality scoring, and industrial recommendations.

    **Dataset:** NEU Surface Defect Dataset  
    **Classes:** Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches  
    **Image size:** 640 × 640 after YOLO preprocessing  
    **Final model:** YOLOv8s  
    **Decision output:** PASS, REVIEW, or REJECT

    **New features included**  
    1. Surface Quality Score out of 100  
    2. Risk Level estimation  
    3. Smart industrial recommendations for each defect  
    4. Inspection ID for traceability  
    5. PDF inspection report export  
    6. Factory analytics dashboard
    7. Video inspection module
    8. Explainability heatmaps and defect-location map
    9. Predictive maintenance insight
    10. Conveyor belt simulation
    11. Operator notes and manual QA review
    12. Critical alert assistant and downloadable supervisor alert template

    **Workflow**  
    1. Upload or capture a surface image  
    2. Run YOLOv8s inference  
    3. Filter detections using the confidence threshold  
    4. Display bounding boxes, defect classes, severity, confidence, and recommendations  
    5. Generate inspection summary, PDF report, and downloadable CSV records
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>VisionAlloy | Automated Metallic Surface Defect Inspection Dashboard</div>", unsafe_allow_html=True)
