import os
import subprocess
import random
from pathlib import Path
from typing import List

# ---- CONFIGURATION ----
OLD_DATASET = "/content/old_dataset"
NEW_DATASET = "/content/new_dataset"
COMBINED_DATASET = "/content/combined_dataset"
PREV_WEIGHTS = "/content/best.pt"
IMAGE_EXTENSIONS = [".jpg", ".jpeg"]
LABEL_EXTENSIONS = [".txt"]
IMAGE_THRESHOLD = 200
LABEL_THRESHOLD = 200
OLD_PERCENTAGE = 0.2
NEW_PERCENTAGE = 0.8
EPOCHS = 50
BATCH_SIZE = 8
IMGSZ = 640
RUN_NAME = "auto_incremental_training"

# ---- UTILITY FUNCTIONS ----
def count_files(directory: str, extensions: List[str]) -> int:
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                count += 1
    return count

def combine_datasets_if_needed():
    # Count new data
    new_img_count = count_files(NEW_DATASET, IMAGE_EXTENSIONS)
    new_lbl_count = count_files(NEW_DATASET, LABEL_EXTENSIONS)
    print(f"New dataset: {new_img_count} images, {new_lbl_count} labels.")

    if new_img_count >= IMAGE_THRESHOLD and new_lbl_count >= LABEL_THRESHOLD:
        print("Threshold met. Combining datasets...")

        # Run Combinderwvalid.py as a subprocess
        cmd = [
            "python", "Combinderwvalid.py",
            "--old-dataset", OLD_DATASET,
            "--new-dataset", NEW_DATASET,
            "--output", COMBINED_DATASET,
            "--old-percentage", str(OLD_PERCENTAGE),
            "--new-percentage", str(NEW_PERCENTAGE)
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    else:
        print(f"Not enough new data to proceed. Need >= {IMAGE_THRESHOLD} images and >= {LABEL_THRESHOLD} labels.")
        return False

def launch_yolo_training():
    data_yaml = os.path.join(COMBINED_DATASET, "data.yaml")
    cmd = [
        "yolo", "detect", "train",
        f"data={data_yaml}",
        f"model={PREV_WEIGHTS}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={BATCH_SIZE}",
        f"name={RUN_NAME}"
    ]
    print(f"Launching YOLOv8 training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# ---- MAIN SCRIPT ----
def main():
    if combine_datasets_if_needed():
        launch_yolo_training()

if __name__ == "__main__":
    main()