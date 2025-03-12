from ultralytics import YOLO
import time

# Load the exported TensorRT model
trt_model = YOLO("yolov11m_400/runs/detect/train/weights/best.engine", task= "detect")

# results = trt_model.predict(source = "../input/video/ad_short.mp4", show =True, save=False, verbose=False, half=True, task = "detect", stream= True)
results = trt_model.predict(source = 0, show =False, save=False, verbose=False, half=True, task = "detect", stream= True)

fps_time = time.perf_counter()

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

    for box in r.boxes.xyxy:  # xyxy format [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box[:4])

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))


    fps_time = time.perf_counter()