import os
import shutil
import subprocess
import random
from pathlib import Path
from typing import List
from yolov8combiner import combine_yolo_datasets
from ultralytics import YOLO
import torch

print(f"GPU available: {torch.cuda.is_available()}")

root="Sidhi-Project\backend\routes\verified"

def ready_for_training(min_images=200):
    return all(
        len(os.listdir(os.path.join(root, cls, "images"))) >= min_images
        for cls in os.listdir(root)
        if os.path.isdir(os.path.join(root, cls))
    )

def rotate_dataset_folders(base_path='Sidhi-Project/Datasets'):
    new_path = os.path.join(base_path, 'New')
    old_path = os.path.join(base_path, 'Old')

    # Delete Old folder
    if os.path.exists(old_path):
        shutil.rmtree(old_path)
        print(f"Deleted: {old_path}")

    # Rename New to Old
    if os.path.exists(new_path):
        os.rename(new_path, old_path)
        print(f"Renamed: {new_path} → {old_path}")
    else:
        print(f"Warning: New folder doesn't exist at {new_path}")

    # Create fresh New folder
    os.makedirs(new_path)
    print(f"Created empty: {new_path}")

if ready_for_training():
    success = combine_yolo_datasets(
        existing_dataset_path="Sidhi-Project\Datasets\Old",
        new_classes_folder=root,
        output_path="Sidhi-Project\Datasets\New",
        existing_percentage=0.2,  # Use 20% of existing dataset
        split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1},  # split ratios
        random_seed=123
    )
    if success:
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    model = YOLO('Sidhi-Project\backend\weights\incrementalv8.pt')
    
    results = model.train(
        data='/content/combinedfixed/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        freeze=10,        # freeze backbone
        lr0=0.001         # slow learning rate
    )
    print(results)
    rotate_dataset_folders()




