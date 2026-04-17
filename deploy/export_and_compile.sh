#!/usr/bin/env bash
# Export YOLOv8 best.pt → TFLite INT8 → Edge TPU, then deploy to models/
#
# Usage (on the Ubuntu VM, from the project root):
#   bash deploy/export_and_compile.sh /home/mac/best.pt
#
# Requirements on VM:
#   - sunba_autotrack venv active (source bin/activate)
#   - pip install ultralytics (use the computer_vision venv or any Python 3.12 venv)
#   - edgetpu_compiler installed (from libedgetpu apt repo)
set -euo pipefail

PT_FILE="${1:-}"
if [[ -z "$PT_FILE" ]]; then
    echo "Usage: bash deploy/export_and_compile.sh /path/to/best.pt"
    exit 1
fi

PT_FILE="$(realpath "$PT_FILE")"
PT_DIR="$(dirname "$PT_FILE")"
PT_STEM="$(basename "$PT_FILE" .pt)"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] Exporting $PT_FILE → TFLite INT8..."
python -c "
from ultralytics import YOLO
YOLO('$PT_FILE').export(format='tflite', int8=True, imgsz=320)
"

TFLITE="$PT_DIR/${PT_STEM}_saved_model/${PT_STEM}_full_integer_quant.tflite"
if [[ ! -f "$TFLITE" ]]; then
    echo "ERROR: TFLite file not found at $TFLITE"
    echo "Check the export output above for the actual path."
    exit 1
fi

echo "[2/3] Compiling for Edge TPU..."
edgetpu_compiler "$TFLITE" --out_dir "$PT_DIR/"

EDGETPU="$PT_DIR/${PT_STEM}_full_integer_quant_edgetpu.tflite"
if [[ ! -f "$EDGETPU" ]]; then
    echo "ERROR: Edge TPU model not found at $EDGETPU"
    exit 1
fi

echo "[3/3] Deploying to $PROJECT_DIR/models/..."
cp "$EDGETPU" "$PROJECT_DIR/models/"

DEPLOYED="$PROJECT_DIR/models/$(basename "$EDGETPU")"
echo ""
echo "Done. Model deployed to: $DEPLOYED"
echo ""
echo "Update config.py:"
echo "  MODEL_TYPE      = \"yolov8\""
echo "  _YOLO_DOG_MODEL = \"models/$(basename "$EDGETPU")\""
