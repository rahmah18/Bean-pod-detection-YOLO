import cv2
import streamlit as st
from ultralytics import YOLO
import os
import numpy as np

def app():
    st.header('Bean Pod Detection Web App')
    st.subheader('Powered by YOLOv8 for Mature Pods Detection')
    st.write('Welcome!')

    # Load your custom-trained model here
    model = YOLO('C:/Users/Admin/Desktop/Model folder/yolo_model_mat/yolov8_mat_pod_model/weights/best.pt')
    object_names = list(model.names.values())  # Should reflect the mature pods class

    with st.form("my_form"):
        uploaded_files = st.file_uploader("Upload images (multiple allowed)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['mature_pod'])  # Your class name
        min_confidence = st.slider('Confidence score', 0.0, 1.0, 0.5)  # Default confidence at 0.5
        st.form_submit_button(label='Submit')

    if uploaded_files:  # Check if any files were uploaded
        for uploaded_file in uploaded_files:  # Loop through each uploaded file
            input_path = uploaded_file.name
            file_binary = uploaded_file.read()
            
            # Save the image temporarily to the disk for processing
            with open(input_path, "wb") as temp_file:
                temp_file.write(file_binary)
            
            image = cv2.imread(input_path)

            with st.spinner(f'Processing {input_path}...'):
                result = model(image)

                # Counter for detected pods in the current image
                pod_count = 0  # Initialize counter for pods detected

                # Iterate through detections and draw bounding boxes
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'
                
                    bbox_thickness = 16  # Thickness of the bounding box

                    # Only consider objects that meet the selected criteria
                    if object_name in selected_objects and score > min_confidence:
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), bbox_thickness)  # Red bounding box
                        cv2.putText(image, label, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        pod_count += 1  # Increment the pod count

                # Convert the image to RGB format for Streamlit display
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(frame, caption=f"Processed {input_path}", channels="RGB")

                # Display the number of detected pods for the current image
                st.write(f"Number of mature pods detected in {input_path}: {pod_count}")

if __name__ == "__main__":
    app()
