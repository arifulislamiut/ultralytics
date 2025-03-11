
"""[![Roboflow Notebooks](https://media.roboflow.com/notebooks/template/bannertest2-2.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672932710194)](https://github.com/roboflow/notebooks)

# How to Train YOLO11 Object Detection on a Custom Dataset

---

[![GitHub](https://badges.aleen42.com/src/github.svg)](https://github.com/ultralytics/ultralytics)

YOLO11 builds on the advancements introduced in YOLOv9 and YOLOv10 earlier this year, incorporating improved architectural designs, enhanced feature extraction techniques, and optimized training methods.

YOLO11m achieves a higher mean mAP score on the COCO dataset while using 22% fewer parameters than YOLOv8m, making it computationally lighter without sacrificing performance.

YOLOv11 is available in 5 different sizes, ranging from `2.6M` to `56.9M` parameters, and capable of achieving from `39.5` to `54.7` mAP on the COCO dataset.

## Setup

**NOTE:** To make it easier for us to manage datasets, images and models we create a `HOME` constant.
"""

import os, subprocess
HOME = os.getcwd()
print(HOME)

"""## Install YOLO11 via Ultralytics"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics supervision roboflow
import ultralytics
ultralytics.checks()
secret_value_0 = "R8JQyQmyXKa0HMr7HzJT"

"""# Fine-tune YOLO11 on custom dataset

**NOTE:** When training YOLOv11, make sure your data is located in `datasets`. If you'd like to change the default location of the data you want to use for fine-tuning, you can do so through Ultralytics' `settings.json`. In this tutorial, we will use one of the [datasets](https://universe.roboflow.com/liangdianzhong/-qvdww) available on [Roboflow Universe](https://universe.roboflow.com/). When downloading, make sure to select the `yolov11` export format.
"""

datadir = f"{HOME}/datasets"
os.makedirs(datadir, exist_ok=True)
from roboflow import Roboflow
rf = Roboflow(api_key=secret_value_0)
project = rf.workspace("ds-gfhaj").project("textile-defect-datasts")
version = project.version(2)
dataset = version.download("yolov11")

"""# Custom Training"""

# Disable WandB integration to prevent logging and visualization during training

os.environ['WANDB_MODE'] = 'disabled'

# Commented out IPython magic to ensure Python compatibility.
# %cd {HOME}

# !yolo task=detect mode=train model=yolo11l.pt data={dataset.location}/data.yaml epochs=700 imgsz=640 plots=True

"""**NOTE:** The results of the completed training are saved in `{HOME}/runs/detect/train/`. Let's examine them."""

import glob
import os
latest_train_folder = max(glob.glob(f'{HOME}/runs/detect/train*/'), key=os.path.getmtime)
print(latest_train_folder)
# !ls {latest_train_folder}

from IPython.display import Image as IPyImage

IPyImage(filename=f'{latest_train_folder}/confusion_matrix.png', width=900)

from IPython.display import Image as IPyImage

IPyImage(filename=f'{latest_train_folder}/results.png', width=900)

from IPython.display import Image as IPyImage

IPyImage(filename=f'{latest_train_folder}/val_batch0_pred.jpg', width=900)

"""# Validate fine-tuned model"""

# !yolo task=detect mode=val model={latest_train_folder}/weights/best.pt data={dataset.location}/data.yaml

"""## Inference with custom model"""

# !yolo task=detect mode=predict model={latest_train_folder}/weights/best.pt conf=0.25 source="/kaggle/working/datasets/Textile-defect-datasts-2/test/images" save=True

"""**NOTE:** Let's take a look at few results.

## Export to TensorRT model
"""

# # Export a YOLO11n PyTorch model to TensorRT format
# !yolo export model={latest_train_folder}/weights/best.pt format=engine  # creates 'yolo11n.engine''

# # Run inference with the exported model
# !yolo predict model=model={latest_train_folder}/weights/best.engine source='https://ultralytics.com/images/bus.jpg'
# print("TRT is valid!")

import glob
import os
from IPython.display import Image as IPyImage, display

latest_pred_folder = max(glob.glob('/kaggle/working/runs/detect/predict*/'), key=os.path.getmtime)
print(latest_pred_folder)
for img in glob.glob(f'{latest_pred_folder}/*.jpg')[:3]:
    display(IPyImage(filename=img, width=600))
    print("\n")

"""# Apply Model to the Video"""

# Input video path for the first video in Kaggle
input_video_path = "/kaggle/input/ad-short/ad_short.mp4"  # First video path

# Output paths for saving the prediction result
output_video_path = latest_pred_folder + "/ad_short_pred.mp4"

# Run YOLO on the first video for object detection
# !yolo task=detect mode=predict model={latest_train_folder}/weights/best.pt conf=0.25 source="{input_video_path}" save=True

latest_pred_folder = max(glob.glob('/kaggle/working/runs/detect/predict*/'), key=os.path.getmtime)
# Output video format is .avi
# !ls {latest_pred_folder}

"""# Convert .avi to .mp4"""

# Convert .avi to .mp4 using FFmpeg
import os

# Path to the input .avi video
input_video = latest_pred_folder + '/ad_short.avi'
# Path to the output .mp4 video
output_video = latest_pred_folder + '/ad_short.mp4'

# FFmpeg command to convert .avi to .mp4
ffmpeg_command = f"ffmpeg -i {input_video} -vcodec libx264 {output_video}"
os.system(ffmpeg_command)

# !ls {latest_pred_folder}

"""# Video with Object Detection applied"""

# Path to the pred video (after conversion to .mp4)

pred_video_path = latest_pred_folder + '/ad_short.mp4'

# Load and encode the video
mp4 = open(pred_video_path, 'rb').read()
data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()

# Embed the video and display it in the notebook
HTML(f"""
<video width=600 controls>
      <source src="{data_url}" type="video/mp4">
</video>
""")

"""# <div style="color:white;border-radius:80px;background-color:green;font-family:Nexa;overflow:hidden"><p style="padding:5px;color:white;text-align:center;overflow:hidden;font-size:100%;letter-spacing:0.5px;margin:0"><b> </b>Thanks to everyone for reviewing this notebook! I would appreciate an upvote if you liked it. Also, I'm curious about your thoughts, so please leave a comment to help me improve.</p></div>"""