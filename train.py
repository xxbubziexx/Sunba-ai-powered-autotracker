"""
YOLOv8 training script — dog detection.

Runs on Windows with GPU. Produces best.pt only.
Export + Edge TPU compilation is handled on the VM by deploy/export_and_compile.sh.

Workflow:
  1. Annotate in Roboflow → export YOLOv8 format → extract into finetuning/
  2. python split_dataset.py      (one-time: creates finetuning/valid/ from train)
  3. python train.py              (trains, saves best.pt)
  4. scp best.pt to VM, run deploy/export_and_compile.sh

Requirements (separate venv from autotrack — ultralytics upgrades numpy):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
  pip install ultralytics
"""

import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_YAML = "finetuning/data.yaml"
BASE_MODEL   = "yolov8s.pt"       # downloaded automatically on first run
IMG_SIZE     = 320                # must match Edge TPU model input
EPOCHS       = 100
BATCH_SIZE   = 8                  # reduce to 4 if still OOM
DEVICE       = "auto"             # auto = GPU if available, else CPU
OUTPUT_NAME  = "dog_detector"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    try:
        import torch
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependencies. Run: pip install ultralytics")
        sys.exit(1)

    if DEVICE == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE
    print(f"Device: {'GPU ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model   = YOLO(BASE_MODEL)
    results = model.train(
        data     = DATASET_YAML,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH_SIZE,
        device   = device,
        name     = OUTPUT_NAME,
        patience = 20,
        save     = True,
        plots    = True,
        workers  = 2,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"\nNext — copy to VM and export:")
    print(f"  scp \"{best}\" mac@<vm-ip>:/home/mac/")
    print(f"  bash deploy/export_and_compile.sh /home/mac/best.pt")


if __name__ == "__main__":
    main()
