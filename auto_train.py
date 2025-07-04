import os
import shutil
import subprocess
import random
from pathlib import Path
from typing import List
from yolov8combiner import combine_yolo_datasets
from ultralytics import YOLO
import torch
from flask import Flask

app = Flask(__name__)


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



def run_autotrain():
    import os, shutil
    from pathlib import Path
    from ultralytics import YOLO

    try:
        if not ready_for_training():
            return {"status": "failed", "message": "Not ready for training yet."}

        success = combine_yolo_datasets(
            existing_dataset_path="Sidhi-Project\\Datasets\\Old",
            new_classes_folder=root,
            output_path="Sidhi-Project\\Datasets\\New",
            existing_percentage=0.2,
            split_ratios={'train': 0.7, 'val': 0.2, 'test': 0.1},
            random_seed=123
        )
        if not success:
            return {"status": "failed", "message": "Dataset combining failed."}

        # Clear new classes folder
        for item in os.listdir(root):
            item_path = os.path.join(root, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        model = YOLO('Sidhi-Project\\backend\\weights\\incrementalv8.pt')

        results = model.train(
            data='Sidhi-Project\\Datasets\\New\\data.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            device=0,
            freeze=10,
            lr0=0.001
        )
        rotate_dataset_folders()

        best_pt_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
        target_dir = Path("Sidhi-Project/backend/weights")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt_path, target_dir / "best.pt")

        return {
            "status": "success",
            "message": "Training completed and best.pt saved.",
            "train_results": str(results),  # you can customize how much info you want here
            "best_pt_path": str(target_dir / "best.pt")
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}




