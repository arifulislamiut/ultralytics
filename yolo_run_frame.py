from ultralytics import YOLO
import argparse
import cv2
import time
import random
import json
from paho.mqtt import client as mqtt_client

broker = 'localhost'
port = 1883
topic = "defect-status"
client_id = f'publish-{random.randint(0, 1000)}'
username = 'jetson'
password = 'mmqtt'

parser = argparse.ArgumentParser(description="Process defect and timestamp.")
parser.add_argument("--show", action="store_true", help="Enable video feed", default=False)
args = parser.parse_args()
show = args.show

print(show)

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, prop):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

# client = connect_mqtt()
# client.loop_start()

# Load the exported TensorRT model
trt_model = YOLO("runs/detect/train24/weights/best.pt", task= "detect")

# # Open the CSI camera (video0)
# cap = cv2.VideoCapture("/dev/video0")
cap = cv2.VideoCapture("inputs/ad_short.mp4")

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()
last_state = False
while True:
    fps_time = time.perf_counter()

    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame!")
        break


    results = trt_model.predict(frame, save = False, verbose = False, half = True, show = False, task = "detect")

    for r in results:
        has_defect = False
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            class_id = int(cls)
            class_name = trt_model.names[class_id]
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if class_name == "defect":
                has_defect = True

        if last_state != has_defect:
            last_state = has_defect

            data = {
                "defect": has_defect,
                "time": time.time()
            }
            msg = json.dumps(data)
            # result = client.publish(topic, msg)

            # status = result[0]
            # if status != 0:
            #     print(f"Failed to send message to topic {topic}")

        last_state = has_defect


    # Display the frame

    fps = 1.0 / (time.perf_counter() - fps_time)
    netFps = "Net FPS: " + str(fps)
    print(netFps)
    cv2.putText(frame, netFps, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if True:
        cv2.imshow('Camera feed', frame)
        cv2.waitKey(20)

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
