# Bean-pod-detection-YOLO
This project applies YOLOv5/YOLOv8/YOLOv11 and OpenCV for real-time bean pod detection and maturity prediction.   The goal is to support farmers and researchers by automating the process of identifying mature bean pods ready for harvest.

# Tools and library used

- Python
- YOLOv5, YOLOv8, YOLOv11
- OpenCV
- PyTorch
- CVAT (annotation)
- jupyter notebook (training)


  #  Workflow
1. **Data Collection & Annotation**  
   - Captured images using HD phone camera.  
   - Labeled pods & plant parts in CVAT.  

2. **Model Training (YOLOv5/v8/v11)**  
   - Trained models on Jupter notebook.  
   - Used augmentation (flip, rotate, color jitter).  
   - Evaluated with mAP, Precision, Recall, F1-score.  

3. **Evaluation**  
   - Best performing model: **YOLOv8**  
   - Achieved **mAP50 = 0.92** and **F1-score = 0.87**.  

4. **Deployment**  
   - Inference pipeline using **OpenCV + YOLO** for real-time detection.  
   - Outputs bounding boxes on pods with confidence scores.  
