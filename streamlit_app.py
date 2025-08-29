
import streamlit as st
from ultralytics import YOLO
import numpy as np
import io
import zipfile
from PIL import Image
import cv2
import time

# Page config
st.set_page_config(
    page_title="🌱 Mature Bean Pod Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🌱 Bean Pod Detection Web App")
st.caption("Powered by YOLOv8 — Detecting mature bean pods with computer vision.")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    return YOLO("Models/yolov8-model/best.pt")

model = load_model()
object_names = list(model.names.values())

# Sidebar controls
st.sidebar.header("⚙️ Settings")
uploaded_files = st.sidebar.file_uploader("Upload bean plant images", 
                                          type=['jpg', 'jpeg', 'png'], 
                                          accept_multiple_files=True)
selected_objects = st.sidebar.multiselect('Objects to detect', object_names, default=object_names)
min_confidence = st.sidebar.slider('Confidence score', 0.0, 1.0, 0.5)
process_btn = st.sidebar.button("🚀 Run Detection")

# Main content
if process_btn and uploaded_files:
    progress = st.progress(0)
    status_text = st.empty()
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for idx, uploaded_file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            status_text.text(f"🔍 Processing {uploaded_file.name}...")
            results = model.predict(image, conf=min_confidence)
            detections = [box for box in results[0].boxes if model.names[int(box.cls)] in selected_objects]
            pod_count = len(detections)

            annotated_img = results[0].plot()

            # Show results
            st.subheader(f"📷 {uploaded_file.name}")
            st.image(annotated_img, channels="BGR", use_column_width=True)
            st.success(f"✅ Detected {pod_count} pods ({', '.join(selected_objects)})")

            # Save annotated image
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_rgb)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="PNG")
            zipf.writestr(f"annotated_{uploaded_file.name}.png", img_bytes.getvalue())

            # Update progress bar
            progress.progress((idx + 1) / len(uploaded_files))

    # Finalize ZIP
    zip_buffer.seek(0)
    status_text.text("🎉 All images processed!")
    st.download_button(
        label="⬇️ Download All Annotated Images (ZIP)",
        data=zip_buffer,
        file_name="bean_pod_detections.zip",
        mime="application/zip"
    )
elif process_btn:
    st.warning("⚠️ Please upload at least one image to continue.")
