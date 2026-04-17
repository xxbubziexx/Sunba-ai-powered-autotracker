"""
YOLOv8n Edge TPU inference wrapper.

Loads a full-integer-quantised YOLOv8n TFLite model compiled for the
Google Coral USB Accelerator via pycoral/tflite-runtime.

Expected model output format (YOLOv8 TFLite export):
  Single output tensor: [1, 84, num_anchors] or [1, num_anchors, 84]
  84 = 4 box coords (cx,cy,w,h) + 80 COCO class scores

Detection namedtuple fields:
  x1, y1, x2, y2  — absolute pixel coords in original frame
  conf             — confidence score [0,1]
  class_id         — COCO class index
"""

import logging
import threading
import queue
import time
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


class Detection(NamedTuple):
    x1:       float
    y1:       float
    x2:       float
    y2:       float
    conf:     float
    class_id: int


class YOLODetector:
    """
    Wraps a YOLOv8n Edge TPU TFLite model.

    Runs inference in a dedicated thread. Consumers call get_detections()
    to retrieve the latest results without blocking the capture thread.

    Usage:
        det = YOLODetector()
        det.start()
        det.push_frame(bgr_frame)
        detections = det.get_detections()
        det.stop()
    """

    def __init__(self):
        self._in_q:   queue.Queue = queue.Queue(maxsize=1)
        self._out_q:  queue.Queue = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._interpreter = None
        self._input_details  = None
        self._output_details = None
        self._input_size: Tuple[int, int] = (config.MODEL_INPUT, config.MODEL_INPUT)

    def start(self):
        self._load_model()
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="detector"
        )
        self._thread.start()
        log.info("Detector started (input %dx%d)", *self._input_size)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def push_frame(self, frame: np.ndarray):
        """Submit a frame for inference. Drops previous frame if not yet consumed."""
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._in_q.put_nowait(frame)
        except queue.Full:
            pass

    def get_detections(self, timeout: float = 0.05) -> Optional[List[Detection]]:
        """
        Returns the latest detection list or None if no result is ready.
        Drains stale results to stay real-time.
        """
        result = None
        try:
            result = self._out_q.get(timeout=timeout)
            while True:
                result = self._out_q.get_nowait()
        except queue.Empty:
            pass
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        """
        Load TFLite model. Tries backends in order:
          1. pycoral (preferred — proper Edge TPU support)
          2. tflite-runtime with Edge TPU delegate DLL
          3. CPU-only fallback
        """
        # ── 1. pycoral ────────────────────────────────────────────────────────
        try:
            from pycoral.utils.edgetpu import make_interpreter
            self._interpreter = make_interpreter(config.MODEL_PATH)
            self._interpreter.allocate_tensors()
            self._input_details  = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            shape = self._input_details[0]["shape"]
            self._input_size = (int(shape[2]), int(shape[1]))
            log.info("Model loaded via pycoral Edge TPU")
            return
        except Exception as exc:
            log.debug("pycoral not available: %s", exc)

        # ── 2. tflite-runtime + Edge TPU delegate DLL ─────────────────────────
        Interpreter = self._get_interpreter_class()
        for dll in ("edgetpu.dll", "libedgetpu.so.1", "libedgetpu.1.dylib"):
            delegate = self._load_delegate(dll)
            if delegate is None:
                continue
            try:
                self._interpreter = Interpreter(
                    model_path=config.MODEL_PATH,
                    experimental_delegates=[delegate],
                )
                log.info("Model loaded with Edge TPU delegate (%s)", dll)
                break
            except Exception as exc:
                log.debug("Delegate %s failed to init interpreter: %s", dll, exc)
        else:
            # ── 3. CPU fallback ───────────────────────────────────────────────
            self._interpreter = Interpreter(model_path=config.MODEL_PATH)
            log.warning("Edge TPU delegate not found — running on CPU")

        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        shape = self._input_details[0]["shape"]
        self._input_size = (int(shape[2]), int(shape[1]))

    @staticmethod
    def _load_delegate(dll_name):
        """Try loading an Edge TPU delegate via tflite_runtime or ai_edge_litert."""
        for module in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
            try:
                mod = __import__(module, fromlist=["load_delegate"])
                return mod.load_delegate(dll_name)
            except Exception:
                pass
        return None

    @staticmethod
    def _get_interpreter_class():
        """Return the best available TFLite Interpreter class."""
        try:
            from tflite_runtime.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        try:
            from ai_edge_litert.interpreter import Interpreter
            return Interpreter
        except ImportError:
            pass
        from tensorflow.lite.python.interpreter import Interpreter
        return Interpreter

    def _inference_loop(self):
        while self._running:
            try:
                frame = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                detections = self._run_inference(frame)
            except Exception as exc:
                log.error("Inference error: %s", exc)
                continue

            if self._out_q.full():
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._out_q.put_nowait(detections)
            except queue.Full:
                pass

    def get_last_input_frame(self) -> Optional[np.ndarray]:
        """Return the last preprocessed model input frame (for debug display)."""
        return getattr(self, "_last_input_frame", None)

    def _run_inference(self, frame: np.ndarray) -> List[Detection]:
        orig_h, orig_w = frame.shape[:2]
        w_in, h_in = self._input_size

        resized = cv2.resize(frame, (w_in, h_in))
        self._last_input_frame = resized.copy()
        inp = self._input_details[0]

        if inp["dtype"] == np.uint8:
            tensor = resized.astype(np.uint8)
        elif inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            tensor = (resized.astype(np.float32) / 255.0 / scale + zero).clip(-128, 127).astype(np.int8)
        else:
            tensor = (resized.astype(np.float32) / 255.0)

        self._interpreter.set_tensor(inp["index"], tensor[np.newaxis])
        self._interpreter.invoke()

        if config.MODEL_TYPE == "yolov8":
            raw = self._interpreter.get_tensor(self._output_details[0]["index"])
            return self._parse_yolov8_output(raw, orig_w, orig_h, w_in, h_in)
        return self._parse_ssd_output(orig_w, orig_h)

    def _parse_yolov8_output(
        self,
        raw: np.ndarray,
        orig_w: int,
        orig_h: int,
        in_w: int,
        in_h: int,
    ) -> List[Detection]:
        """YOLOv8 TFLite output: single tensor [1, 84, N] or [1, N, 84]."""
        out = raw[0]
        # Transpose if shape is [features, anchors] rather than [anchors, features]
        if out.shape[0] < out.shape[1]:
            out = out.T

        detail = self._output_details[0]
        if detail["dtype"] in (np.uint8, np.int8):
            scale, zero = detail["quantization"]
            out = (out.astype(np.float32) - zero) * scale

        boxes  = out[:, :4]
        scores = out[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confs     = scores[np.arange(len(scores)), class_ids]

        detections = []
        for i, (conf, cls_id) in enumerate(zip(confs, class_ids)):
            if conf < config.CONF_THRESH:
                continue
            cx, cy, bw, bh = boxes[i]
            x1 = max(0.0, min(float((cx - bw / 2) * orig_w), orig_w))
            y1 = max(0.0, min(float((cy - bh / 2) * orig_h), orig_h))
            x2 = max(0.0, min(float((cx + bw / 2) * orig_w), orig_w))
            y2 = max(0.0, min(float((cy + bh / 2) * orig_h), orig_h))
            detections.append(Detection(x1, y1, x2, y2, float(conf), int(cls_id)))

        return self._nms(detections) if detections else detections

    def _parse_ssd_output(self, orig_w: int, orig_h: int) -> List[Detection]:
        """
        Parse TFLite Detection API output (SSD MobileNet V2 style).
        4 output tensors:
          0: boxes   [1, N, 4]  — [ymin, xmin, ymax, xmax] normalised
          1: classes [1, N]     — class index (float, 0-based, no background)
          2: scores  [1, N]     — confidence scores
          3: count   [1]        — number of valid detections
        """
        def _dequant(idx):
            t = self._interpreter.get_tensor(self._output_details[idx]["index"])
            d = self._output_details[idx]
            if d["dtype"] in (np.uint8, np.int8):
                scale, zero = d["quantization"]
                t = (t.astype(np.float32) - zero) * scale
            return t

        boxes   = _dequant(0)[0]   # [N, 4]
        classes = _dequant(1)[0]   # [N]
        scores  = _dequant(2)[0]   # [N]
        count   = int(_dequant(3)[0])

        detections = []
        for i in range(count):
            conf   = float(scores[i])
            cls_id = int(classes[i])

            if conf < config.CONF_THRESH:
                continue

            ymin, xmin, ymax, xmax = boxes[i]
            x1 = max(0.0, float(xmin) * orig_w)
            y1 = max(0.0, float(ymin) * orig_h)
            x2 = min(float(orig_w), float(xmax) * orig_w)
            y2 = min(float(orig_h), float(ymax) * orig_h)

            detections.append(Detection(x1, y1, x2, y2, conf, cls_id))

        return detections

    @staticmethod
    def _nms(detections: List[Detection], iou_thresh: float = 0.45) -> List[Detection]:
        detections = sorted(detections, key=lambda d: d.conf, reverse=True)
        kept = []
        for det in detections:
            overlap = False
            for k in kept:
                if _iou(det, k) > iou_thresh:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        return kept


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return str(class_id)
