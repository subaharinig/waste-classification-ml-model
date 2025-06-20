from ultralytics import YOLO

model = YOLO("garbage_cls/yolov8n_handheld2/weights/best.pt")  # Update path if needed
result = model(r"C:\suba_work\waste-classification-model-main\waste-classification-model-main\static\img1.jpeg")
pred_class = result[0].probs.top1
print("Prediction:", "Recyclable" if pred_class == 0 else "Non-Recycable")
