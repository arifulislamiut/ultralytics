from ultralytics import YOLO

weights_path = "runs/detect/train24/weights/"
# Load a YOLO11n PyTorch model
model = YOLO(f"{weights_path}best.pt")

# # Export the model to TensorRT
model.export(format="engine")  # creates 'yolo11n.engine'