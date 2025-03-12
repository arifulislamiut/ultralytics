# from inference import InferencePipeline
# from inference.core.interfaces.camera.entities import VideoFrame
#
# # import opencv to display our annotated images
# import cv2
# # import supervision to help visualize our predictions
# import supervision as sv
#
# # create a bounding box annotator and label annotator to use in our custom sink
# label_annotator = sv.LabelAnnotator()
# box_annotator = sv.BoxAnnotator()
#
# def my_custom_sink(predictions: dict, video_frame: VideoFrame):
#     # get the text labels for each prediction
#     labels = [p["class"] for p in predictions["predictions"]]
#     # load our predictions into the Supervision Detections api
#     detections = sv.Detections.from_inference(predictions)
#     # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
#     image = label_annotator.annotate(
#         scene=video_frame.image.copy(), detections=detections, labels=labels
#     )
#     image = box_annotator.annotate(image, detections=detections)
#     # display the annotated image
#     cv2.imshow("Predictions", image)
#     cv2.waitKey(1)
#
# pipeline = InferencePipeline.init(
#     model_id="fabric-pzimg/1",
#     video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
#     on_prediction=my_custom_sink,
# )
#
# pipeline.start()
# pipeline.join()

# Import the InferencePipeline object
# from inference import InferencePipeline
# # Import the built in render_boxes sink for visualizing results
# from inference.core.interfaces.stream.sinks import render_boxes
#
# # initialize a pipeline object
# pipeline = InferencePipeline.init(
#     model_id="rock-paper-scissors-sxsw/11", # Roboflow model to use
#     video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
#     on_prediction=render_boxes, # Function to run after each prediction
# )
# pipeline.start()
# pipeline.join()

# from inference import get_model
# model = get_model("rock-paper-scissors-sxsw/11", api_key="R8JQyQmyXKa0HMr7HzJT")
# image = "https://source.roboflow.com/K4ntFFA4D1bj9XbKV2meKDT4t4Y2/LqsjV7ajehzL77gpTOCb/original.jpg" # or PIL.Image or numpy array
# results = model.infer(image)[0]

# import roboflow
#
# rf = roboflow.Roboflow(api_key="R8JQyQmyXKa0HMr7HzJT")
# model = rf.workspace("project-lp5e2").project("fabric-pzimg").version("1").model
# prediction = model.download()

from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv

# create a bounding box annotator and label annotator to use in our custom sink
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()