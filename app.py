import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import io
import zipfile
from PIL import Image

def app():
    st.header('🌱 Bean Pod Detection Web App')
    st.subheader('Powered by YOLOv8 for Mature Pods Detection')
    st.write('Upload bean plant images and detect mature pods.')

    # Load YOLO model (model must be in /models/best.pt inside repo)
    @st.cache_resource
    def load_model():
        return YOLO("Models/yolov8-model/best.pt")
    
    model = load_model()
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_files = st.file_uploader("Upload images (multiple allowed)", 
                                          type=['jpg', 'jpeg', 'png'], 
                                          accept_multiple_files=True)
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=object_names)
        min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)
        submitted = st.form_submit_button(label='Submit')

    if submitted and uploaded_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for uploaded_file in uploaded_files:
                # Read file into memory
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

                with st.spinner(f'Processing {uploaded_file.name}...'):
                    results = model.predict(image, conf=min_confidence)

                    detections = [box for box in results[0].boxes if model.names[int(box.cls)] in selected_objects]
                    pod_count = len(detections)

                    annotated_img = results[0].plot()

                    # Show results
                    st.subheader(f"Results for {uploaded_file.name}")
                    st.image(annotated_img, channels="BGR")
                    st.write(f"✅ Detected {pod_count} {', '.join(selected_objects)}")

                    # Convert for saving
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(annotated_rgb)

                    img_bytes = io.BytesIO()
                    pil_img.save(img_bytes, format="PNG")
                    zipf.writestr(f"annotated_{uploaded_file.name}.png", img_bytes.getvalue())

        # Finalize ZIP
        zip_buffer.seek(0)
        st.download_button(
            label="⬇️ Download All Annotated Images (ZIP)",
            data=zip_buffer,
            file_name="bean_pod_detections.zip",
            mime="application/zip"
        )
