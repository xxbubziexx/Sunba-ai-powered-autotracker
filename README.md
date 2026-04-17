# Sunba Autotrack

AI-powered PTZ autotracking for Sunba (XMEye/Sofia protocol) cameras using a Google Coral USB Accelerator.

Replaces the camera's limited onboard tracking with a fully configurable Python pipeline:
- **Detection** — SSD MobileNet V2 or custom YOLOv8 model running on the Coral Edge TPU
- **Tracking** — proportional PTZ control via the XMEye/DVRIP protocol (port 34567)
- **Patrol** — automated preset cycling when no target is detected
- **Schedule** — time-based pause/resume (e.g. overnight)
- **Dataset capture** — saves frames on target acquisition for model retraining

---

## Hardware

| Component | Notes |
|---|---|
| PTZ camera | Tested on Sunba 405-D20X ECO. Any XMEye/Sofia camera should work |
| Google Coral USB Accelerator | Required for Edge TPU inference |
| Python 3.9 | pycoral only supports up to 3.9 |

---

## Setup

### 1. Python environment (Python 3.9)

```bash
python3.9 -m venv venv39
source venv39/bin/activate        # Linux
venv39\Scripts\activate           # Windows

pip install opencv-python "numpy<2.0" ffpyplayer
```

### 2. Coral Edge TPU runtime

Install the system library from [coral.ai](https://coral.ai/docs/accelerator/get-started/):

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install libedgetpu1-std
```

### 3. pycoral wheels

Download from [github.com/google-coral/pycoral/releases/tag/v2.0.0](https://github.com/google-coral/pycoral/releases/tag/v2.0.0):

```bash
# Linux x86_64
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl
```

### 4. Models

Download from the [Coral model zoo](https://coral.ai/models/object-detection/):

```bash
mkdir models
# SSD MobileNet V2 (recommended starting point — person, cat, dog)
wget https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite -P models/
```

### 5. Configuration

Edit `config.py`:

```python
CAMERA_IP       = "192.168.x.x"   # your camera's IP
CAMERA_USER     = "admin"
CAMERA_PASSWORD = "your_password"
```

Set `MODEL_TYPE = "ssd"` for SSD MobileNet V2 or `"yolov8"` for a custom YOLOv8 model.

---

## Running

```bash
# With display (desktop)
python main.py

# Headless (server / Proxmox VM)
python main.py --headless
```

### Keyboard controls (windowed mode)

| Key | Action |
|---|---|
| Q | Quit |
| P | Pause / resume tracking |
| R | Return to home preset |
| D | Toggle model input debug view |

---

## Patrol

Set patrol presets in `config.py`:

```python
PATROL_PRESETS = [1, 2, 3, 4, 5]  # preset numbers to cycle through
PATROL_DWELL_S = 8.0               # seconds at each preset
```

When a target is detected at preset 3, tracking activates. On target loss the camera returns to preset 3 and resumes patrol from there.

---

## Schedule

```python
SCHEDULE_ENABLED      = True
SCHEDULE_PAUSE_TIME   = "23:00"   # stop tracking, go to preset
SCHEDULE_RESUME_TIME  = "06:00"   # resume patrol
SCHEDULE_PAUSE_PRESET = 1
```

---

## Training a custom model

### 1. Collect data

Enable `DATASET_ENABLED = True` in `config.py`. The system saves one frame per detection event to `dataset/`.

### 2. Annotate

Upload `dataset/` to [Roboflow](https://roboflow.com), annotate bounding boxes, export as **YOLOv8 format** into `finetuning/`.

### 3. Split and train (Windows with GPU)

```bash
python split_dataset.py   # creates finetuning/valid/ from train set
python train.py           # produces runs/detect/dog_detector/weights/best.pt
```

### 4. Export and compile (Linux VM)

```bash
scp best.pt user@vm:/home/user/
bash deploy/export_and_compile.sh /home/user/best.pt
```

The compiled `_edgetpu.tflite` is automatically copied to `models/`.

### 5. Switch model

```python
MODEL_TYPE      = "yolov8"
_YOLO_DOG_MODEL = "models/best_full_integer_quant_edgetpu.tflite"
```

---

## Proxmox / Server deployment

See `deploy/` for:
- `setup_vm.sh` — Ubuntu 24.04 VM setup script
- `sunba-autotrack.service` — systemd unit for 24/7 operation
- `99-coral-usb.rules` — udev rules to prevent Coral USB autosuspend
- `export_and_compile.sh` — Edge TPU model export pipeline

Run the service:

```bash
sudo cp deploy/sunba-autotrack.service /etc/systemd/system/
sudo systemctl enable --now sunba-autotrack
journalctl -fu sunba-autotrack
```

---

## Protocol

The camera is controlled via the **XMEye/Sofia/DVRIP** protocol on TCP port 34567 — the same protocol used by the web UI and mobile apps. No ONVIF or RTSP PTZ required.

PTZ movement requires continuous command sending (~5 Hz heartbeat). The `PTZController` class handles this automatically.
