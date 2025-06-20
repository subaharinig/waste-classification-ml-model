from ultralytics import YOLO

# Path to the classification dataset
data_dir = 'smart_bin_dataset'

# Load YOLOv8 classification model (yolov8n-cls.pt is smallest)
model = YOLO('yolov8n-cls.pt')  # You can also use yolov8s-cls.pt for better performance

# Train the model
model.train(
    data=data_dir,
    epochs=30,         # Increase if you have more data
    imgsz=224,         # Image size (224x224 is default for classification)
    project='garbage_cls',
    name='yolov8n_handheld',
    batch=16,          # Adjust according to your RAM/GPU
    patience=5,        # Early stopping if no improvement
)

# Best model is saved automatically in:
# garbage_cls/yolov8n_handheld/weights/best.pt

print("Model training complete. Best model saved at:")
print("garbage_cls/yolov8n_handheld/weights/best.pt")
