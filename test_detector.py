"""
Quick test: load the Edge TPU model and run inference on a blank frame.
Confirms the Coral USB is working before running main.py.
"""

import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

import config
from detector import YOLODetector

print("Starting detector...")
det = YOLODetector()
det.start()

import time
time.sleep(1)

print("Pushing blank frame...")
blank = np.zeros((448, 800, 3), dtype=np.uint8)
det.push_frame(blank)

print("Waiting for inference (first run can take a few seconds)...")
result = det.get_detections(timeout=60.0)
if result is None:
    print("ERROR: No result returned within 60 seconds")
else:
    print(f"OK — inference returned {len(result)} detections on blank frame (expected 0)")

det.stop()
print("Done.")
