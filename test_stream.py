"""
Stream latency diagnostic.
Displays live feed with system clock overlay and FPS/inter-frame stats.
"""

import logging
import time
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from stream import StreamCapture

sc = StreamCapture()
sc.start()

print("Waiting for first frame...")
frame_count = 0
start = None
intervals = []
prev_time = None

try:
    while True:
        result = sc.get_frame(timeout=1.0)
        if result is None:
            continue

        frame, pts = result
        now = time.time()

        if start is None:
            start = now
            print(f"First frame: {(now - start)*1000:.0f}ms after connect")

        if prev_time is not None:
            intervals.append((now - prev_time) * 1000)
        prev_time = now
        frame_count += 1

        bgr = frame.copy()
        h, w = bgr.shape[:2]

        now_str = datetime.now().strftime("%H:%M:%S.%f")[:-4]
        cv2.putText(bgr, now_str, (w // 2 - 120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5)
        cv2.putText(bgr, now_str, (w // 2 - 120, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)

        elapsed = now - start
        fps_str = f"FPS: {frame_count/elapsed:.1f}  Frame: {frame_count}" if elapsed > 0 else f"Frame: {frame_count}"
        if len(intervals) >= 5:
            avg_ms = sum(intervals[-30:]) / len(intervals[-30:])
            std_ms = (sum((x - avg_ms)**2 for x in intervals[-30:]) / len(intervals[-30:])) ** 0.5
            fps_str += f"  Interval avg={avg_ms:.0f}ms std={std_ms:.0f}ms"

        cv2.putText(bgr, fps_str, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Stream Test [Q=quit]", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    sc.stop()
    cv2.destroyAllWindows()
    print("Done.")
