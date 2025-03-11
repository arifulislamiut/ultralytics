import os
import glob
import subprocess
import ultralytics
import matplotlib.pyplot as plt
import time

# Set up home directory
HOME = os.getcwd()
print(f"Working Directory: {HOME}")

datadir = os.path.join(HOME, "datasets")
os.makedirs(datadir, exist_ok=True)

# Install YOLO11 if not installed
try:
    import ultralytics
except ImportError:
    subprocess.run(["pip", "install", "ultralytics", "supervision", "roboflow"], check=True)

# Check installation
ultralytics.checks()

# Load dataset using Roboflow
from roboflow import Roboflow

SECRET_API_KEY = "R8JQyQmyXKa0HMr7HzJT"  # Replace with your Roboflow API Key
rf = Roboflow(api_key=SECRET_API_KEY)
project = rf.workspace("ds-gfhaj").project("textile-defect-datasts")
version = project.version(2)
dataset = version.download("yolov11")

os.environ['WANDB_MODE'] = 'disabled'  # Disable WandB logging

# Hyperparameters
epochs = 700
img_size = 640
batch_size = 16  # Adjust batch size
learning_rate = 0.01  # Adjust learning rate
momentum = 0.937  # Adjust momentum
weight_decay = 0.0005  # Adjust weight decay


def train_yolo():
    """Train YOLOv11 on the dataset with adjustable hyperparameters and real-time loss visualization"""
    command = [
        "yolo", "task=detect", "mode=train", "model=yolo11l.pt",
        f"data={dataset.location}/data.yaml", f"epochs={epochs}",
        f"imgsz={img_size}", f"batch={batch_size}",
        f"lr0={learning_rate}", f"momentum={momentum}",
        f"weight_decay={weight_decay}", "plots=True"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    loss_values = []

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Progress")

    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        if "loss:" in line.lower():
            parts = line.strip().split()
            try:
                loss_index = parts.index("loss:") + 1
                loss = float(parts[loss_index])
                loss_values.append(loss)
                ax.plot(loss_values, marker='o', linestyle='-', color='b')
                plt.draw()
                plt.pause(0.1)
            except (ValueError, IndexError):
                pass

    process.wait()
    plt.ioff()
    plt.show()


def get_latest_train_folder():
    """Find latest training run folder"""
    return max(glob.glob(f'{HOME}/runs/detect/train*/'), key=os.path.getmtime)


def plot_results():
    """Display training loss, confusion matrix, and mAP results using matplotlib"""
    latest_train_folder = get_latest_train_folder()

    # Plot training results
    results_path = os.path.join(latest_train_folder, "results.png")
    cm_path = os.path.join(latest_train_folder, "confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if os.path.exists(results_path):
        results_img = plt.imread(results_path)
        axes[0].imshow(results_img)
        axes[0].axis('off')
        axes[0].set_title("Training Results")

    if os.path.exists(cm_path):
        cm_img = plt.imread(cm_path)
        axes[1].imshow(cm_img)
        axes[1].axis('off')
        axes[1].set_title("Confusion Matrix")

    plt.show()
    print("Training completed. Check results folder for more insights.")


def validate_model():
    """Validate trained YOLO model"""
    latest_train_folder = get_latest_train_folder()
    best_model = os.path.join(latest_train_folder, "weights", "best.pt")
    command = ["yolo", "task=detect", "mode=val", f"model={best_model}", f"data={dataset.location}/data.yaml"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    train_yolo()
    plot_results()
    validate_model()
