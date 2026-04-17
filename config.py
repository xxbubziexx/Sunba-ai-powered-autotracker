"""
Central configuration for the Sunba autotrack system.
All tunable constants live here — do not hardcode values in other modules.
"""

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_IP       = "192.168.x.x"       # your camera's IP address
CAMERA_USER     = "admin"             # XMEye username
CAMERA_PASSWORD = "your_password"    # XMEye password

# RTSP sub-stream (stream=1 → lower resolution, low latency)
RTSP_URL = (
    f"rtsp://{CAMERA_USER}:{CAMERA_PASSWORD}@{CAMERA_IP}:554"
    f"/user={CAMERA_USER}&password={CAMERA_PASSWORD}&channel=1&stream=1.sdp"
)

# ── PTZ ───────────────────────────────────────────────────────────────────────
PTZ_PORT         = 34567
PTZ_CHANNEL      = 0        # camera channel index (0-based)
HOME_PRESET      = 1        # preset number to return to when target is lost
PTZ_KEEPALIVE_S  = 20       # seconds between keepalive pings to hold TCP session

# Speed ranges accepted by the camera (1–8 typical for XMEye)
PTZ_PAN_SPEED_MIN  = 1
PTZ_PAN_SPEED_MAX  = 6
PTZ_TILT_SPEED_MIN = 1
PTZ_TILT_SPEED_MAX = 4
PTZ_ZOOM_SPEED     = 3      # fixed zoom speed (1 = slowest, most precise)

# ── Stream ────────────────────────────────────────────────────────────────────
STREAM_QUEUE_SIZE = 2       # max frames buffered; oldest dropped to stay real-time

# ffpyplayer low-latency options
FF_OPTS = {
    "an":        True,      # disable audio
    "sync":      "video",
    "framedrop": True,
    "infbuf":    False,
}

LIB_OPTS = {
    "rtsp_transport": "tcp",
    "fflags":         "nobuffer",
    "flags":          "low_delay",
    "max_delay":      "100000",
    "timeout":        "3000000",
    "threads":        "auto",
}

# ── Detection ─────────────────────────────────────────────────────────────────
# MODEL_TYPE: "ssd"    → SSD MobileNet V2 (Detection API, 4-tensor output)
#             "yolov8" → YOLOv8 TFLite (single tensor output)
MODEL_TYPE = "ssd"

# SSD MobileNet V2 — person + animals, runs ~100% on Edge TPU
_SSD_MODEL        = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
_SSD_INPUT        = 300
_SSD_CLASSES      = {0, 15, 16}       # person, cat, dog

# YOLOv8s dog-detection — single class (0=dog), fine-tuned on Open Images V7
_YOLO_DOG_MODEL   = "models/best_full_integer_quant_edgetpu.tflite"
_YOLO_DOG_INPUT   = 320
_YOLO_DOG_CLASSES = {0}               # dog only
_YOLO_DOG_LABELS  = {0: "dog"}

# Active model settings resolved from MODEL_TYPE above
if MODEL_TYPE == "yolov8":
    MODEL_PATH     = _YOLO_DOG_MODEL
    MODEL_INPUT    = _YOLO_DOG_INPUT
    TARGET_CLASSES = _YOLO_DOG_CLASSES
    CLASS_LABELS   = _YOLO_DOG_LABELS   # override COCO names for this model
else:
    MODEL_PATH     = _SSD_MODEL
    MODEL_INPUT    = _SSD_INPUT
    TARGET_CLASSES = _SSD_CLASSES
    CLASS_LABELS   = {}                 # empty = use default COCO names

CONF_THRESH = 0.25          # minimum detection confidence to consider

# ── Tracker ───────────────────────────────────────────────────────────────────
AUTOTRACK_ENABLED = True    # set True to enable PTZ tracking

# Deadzone: fraction of frame dimension; no PTZ issued inside this band
DEADZONE_X = 0.12           # ±12% of frame width
DEADZONE_Y = 0.12           # ±12% of frame height

# Proportional gain: maps normalised offset [-1,1] → PTZ speed [1,5]
KP_PAN  = 5.0
KP_TILT = 3.5

# Seconds to wait after issuing a PTZ command before sending a new one
# Prevents oscillation caused by detection latency
PTZ_CMD_COOLDOWN_S = 0.35

# Detection must be continuously present for this long before PTZ activates
DETECTION_STABLE_S = 0

# Consecutive detection frames required before target is considered confirmed
DETECTION_CONFIRM_FRAMES = 1

# Minimum confidence advantage a new target needs to displace a locked target
LOCK_SWITCH_MARGIN = 0.20   # new target must be this much more confident to take over

# Zoom thresholds: bbox area as fraction of total frame area
ZOOM_IN_THRESH  = 0.05      # zoom in  when bbox area < 5%  of frame
ZOOM_OUT_THRESH = 0.25      # zoom out when bbox area > 25% of frame

# Seconds to zoom out for after target is lost (camera returns to wide FOV)
ZOOM_OUT_ON_LOSS_S = 4.0

# Force zoom out when target centre is this far from frame centre (normalised)
# Prevents losing fast targets at the edge when zoomed in
ZOOM_OUT_EDGE_THRESH = 0.40  # zoom out earlier to keep fast targets in frame

# Lost-target: frames without a detection before returning to home preset
LOST_FRAMES_LIMIT = 30          # frames without detection before returning home

# ── Dataset capture ───────────────────────────────────────────────────────────
DATASET_ENABLED = True
DATASET_DIR     = "dataset"       # relative to project root; created automatically

# ── Schedule ──────────────────────────────────────────────────────────────────
SCHEDULE_ENABLED      = True
SCHEDULE_PAUSE_TIME   = "23:00"   # HH:MM — stop tracking/patrol, go to preset
SCHEDULE_RESUME_TIME  = "06:00"   # HH:MM — resume patrol + tracking
SCHEDULE_PAUSE_PRESET = 1         # preset to hold at during scheduled pause

# ── Patrol ────────────────────────────────────────────────────────────────────
PATROL_ENABLED = True
PATROL_PRESETS = [1, 2, 3, 4, 5]   # ordered list; camera cycles through these when idle
PATROL_DWELL_S = 8.0               # seconds to stay at each preset before advancing
