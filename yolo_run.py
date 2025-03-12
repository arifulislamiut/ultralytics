from ultralytics import YOLO

# trt_model = YOLO("runs/detect/train6/weights/best.engine", task= "detect")
trt_model = YOLO("runs/detect/train6/weights/best.pt", task= "detect")
results = trt_model.predict(source = 0, show =True, save=True, verbose=True, half=True, task = "detect")

