"""
Sunba PTZ Autotrack — main entry point.

Starts all threads, runs the display loop, handles keyboard input.

Modes:
  Default   — OpenCV window with overlay (Windows / desktop Linux)
  --headless — No display; SIGTERM/SIGINT for clean shutdown (server/Proxmox)

Keyboard controls (default mode, focus the video window):
  Q — quit
  P — pause / resume tracking
  R — return to home preset immediately
  D — toggle model input debug view
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import config
from detector import YOLODetector, class_name
from ptz import PTZController
from stream import StreamCapture
from tracker import AutoTracker

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOW_TITLE  = "Sunba Autotrack [Q=quit  P=pause  R=home  D=debug]"
WINDOW_DEBUG  = "Model Input (320x320)"

COL_BOX    = (0,   255,   0)
COL_CROSS  = (0,   200, 255)
COL_TEXT   = (255, 255, 255)
COL_WARN   = (0,   80,  255)
COL_PAUSED = (0,   165, 255)


def save_dataset_frame(frame: np.ndarray):
    """Save a raw frame to the dataset directory on target acquisition."""
    if not config.DATASET_ENABLED:
        return
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(config.DATASET_DIR, f"{ts}.jpg")
    cv2.imwrite(path, frame)
    log.info("Dataset: saved %s", path)


def draw_overlay(
    frame:        np.ndarray,
    tracker:      AutoTracker,
    detector:     YOLODetector,
    stream:       StreamCapture,
    paused:       bool,
    lost_count:   int,
    detections=None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    out  = frame.copy()

    cx, cy = w // 2, h // 2
    cv2.line(out, (cx - 20, cy), (cx + 20, cy), COL_CROSS, 1)
    cv2.line(out, (cx, cy - 20), (cx, cy + 20), COL_CROSS, 1)

    # Draw all raw detections when autotrack is off (tuning mode)
    if not config.AUTOTRACK_ENABLED and detections:
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            in_target = det.class_id in config.TARGET_CLASSES
            colour = COL_BOX if in_target else COL_WARN
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            label = f"{config.CLASS_LABELS.get(det.class_id, class_name(det.class_id))} {det.conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), colour, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Draw active tracking target (green box + line to centre)
    target = tracker.active_target
    if target is not None:
        x1, y1, x2, y2 = int(target.x1), int(target.y1), int(target.x2), int(target.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_BOX, 2)
        label = f"{config.CLASS_LABELS.get(target.class_id, class_name(target.class_id))} {target.conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), COL_BOX, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        tcx = (x1 + x2) // 2
        tcy = (y1 + y2) // 2
        cv2.line(out, (cx, cy), (tcx, tcy), COL_CROSS, 1, cv2.LINE_AA)

    status_lines = [
        f"Stream: {'OK' if stream.connected else 'DISCONNECTED'}",
        f"Lost: {lost_count}/{config.LOST_FRAMES_LIMIT}",
        f"Track: {'ON' if config.AUTOTRACK_ENABLED else 'OFF'}",
    ]
    if paused:
        status_lines.insert(0, "PAUSED")
    for i, line in enumerate(status_lines):
        colour = COL_PAUSED if paused and i == 0 else COL_TEXT
        cv2.putText(out, line, (8, h - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)

    dz_x = int(w * config.DEADZONE_X)
    dz_y = int(h * config.DEADZONE_Y)
    cv2.rectangle(
        out,
        (cx - dz_x, cy - dz_y),
        (cx + dz_x, cy + dz_y),
        (80, 80, 80), 1,
    )

    return out


def _shared_startup():
    """Connect PTZ, start stream, wait for first frame, start detector + tracker."""
    ptz = PTZController()
    if not ptz.connect():
        log.error("Cannot connect to PTZ — check camera IP and port. Exiting.")
        sys.exit(1)

    stream   = StreamCapture()
    detector = YOLODetector()

    stream.start()
    log.info("Waiting for first frame to determine resolution...")
    first    = None
    deadline = time.time() + 15
    while first is None and time.time() < deadline:
        first = stream.get_frame(timeout=0.5)
    if first is None:
        log.error("No frame received within 15s — check RTSP URL. Exiting.")
        stream.stop()
        ptz.disconnect()
        sys.exit(1)

    frame_h, frame_w = first[0].shape[:2]
    log.info("Frame resolution: %dx%d", frame_w, frame_h)

    detector.start()
    tracker = AutoTracker(ptz, frame_w=frame_w, frame_h=frame_h)
    tracker.start()

    return ptz, stream, detector, tracker


def _shared_teardown(ptz, stream, detector, tracker):
    log.info("Shutting down...")
    tracker.stop()
    detector.stop()
    stream.stop()
    ptz.disconnect()
    log.info("Done.")


def run_headless():
    """Headless mode: tracking only, no display. Exits cleanly on SIGTERM/SIGINT."""
    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log.info("Signal %s received — stopping.", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    ptz, stream, detector, tracker = _shared_startup()
    log.info("Headless mode running. Send SIGTERM or Ctrl+C to stop.")

    prev_target = None
    try:
        while not stop_event.is_set():
            result = stream.get_frame(timeout=0.1)
            if result is None:
                continue
            frame, _ = result

            detector.push_frame(frame)

            detections = detector.get_detections(timeout=0.0)
            if detections is not None and config.AUTOTRACK_ENABLED:
                tracker.update(detections)

            current_target = tracker.active_target
            if current_target is not None and prev_target is None:
                save_dataset_frame(frame)
            prev_target = current_target
    finally:
        _shared_teardown(ptz, stream, detector, tracker)


def run_windowed():
    """Default windowed mode with OpenCV overlay and keyboard controls."""
    ptz, stream, detector, tracker = _shared_startup()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    paused     = False
    debug_view = False
    prev_target = None

    log.info("Running. Press Q to quit, P to pause, R to home, D to debug.")

    try:
        while True:
            result = stream.get_frame(timeout=0.1)
            if result is None:
                continue
            frame, _ = result

            detector.push_frame(frame)

            detections = detector.get_detections(timeout=0.0)
            if detections is not None and config.AUTOTRACK_ENABLED:
                tracker.update(detections)

            current_target = tracker.active_target
            if current_target is not None and prev_target is None:
                save_dataset_frame(frame)
            prev_target = current_target

            display = draw_overlay(
                frame, tracker, detector, stream,
                paused=paused,
                lost_count=tracker._lost_count,
                detections=detections,
            )
            cv2.imshow(WINDOW_TITLE, display)

            if debug_view:
                inp = detector.get_last_input_frame()
                if inp is not None:
                    cv2.imshow(WINDOW_DEBUG, cv2.resize(inp, (480, 480)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("p"):
                paused = not paused
                tracker.pause(paused)
            elif key == ord("r"):
                tracker.reset_home()
            elif key == ord("d"):
                debug_view = not debug_view
                if not debug_view:
                    try:
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except Exception:
                        pass
                log.info("Debug view: %s", "ON" if debug_view else "OFF")

    finally:
        _shared_teardown(ptz, stream, detector, tracker)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Sunba PTZ Autotrack")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (for server/Proxmox deployment)",
    )
    args = parser.parse_args()

    log.info("Starting Sunba Autotrack (%s)", "headless" if args.headless else "windowed")

    if args.headless:
        run_headless()
    else:
        run_windowed()


if __name__ == "__main__":
    main()
