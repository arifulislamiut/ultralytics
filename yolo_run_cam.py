from ultralytics import YOLO
import time
import random
from paho.mqtt import client as mqtt_client

broker = 'localhost'
port = 1883
topic = "python/mqtt"
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 1000)}'
username = 'jetson'
password = 'mmqtt'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION1, client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def publish(client):
    msg_count = 1
    while True:
        time.sleep(1)
        msg = f"messages: {msg_count}"
        result = client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1
        if msg_count > 5:
            break


# client = connect_mqtt()
# client.loop_start()

# Load the exported TensorRT model
trt_model = YOLO("runs/detect/train24/weights/best.engine", task= "detect")
results = trt_model.track(source = 0, show =True, save=True, verbose=False, half=True, task = "detect", stream= True)
# results = trt_model.track(source="https://youtu.be/LNwODJXcvt4", show=True)

fps_time = time.perf_counter()

for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

    fps = 1.0 / (time.perf_counter() - fps_time)
    print("Net FPS: %f" % (fps))


    fps_time = time.perf_counter()


