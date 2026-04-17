"""
AutoTracker — target selection, PTZ centering, zoom control, lost-target handling.

Control loop (called once per detection result):
  1. Select the highest-confidence detection from target classes.
  2. Compute normalised offset of bbox centre from frame centre [-1, 1].
  3. Apply deadzone — no command if offset is within threshold.
  4. Map offset → PTZ speed using proportional gain (KP).
  5. Issue zoom command based on bbox area relative to frame.
  6. If no detection for LOST_FRAMES_LIMIT consecutive frames → goto home preset.
"""

import logging
import threading
import time
from datetime import datetime, time as dtime
from typing import List, Optional

import config
from detector import Detection
from ptz import PTZController

log = logging.getLogger(__name__)


class AutoTracker:
    """
    Runs the tracking control loop in a dedicated thread.

    Usage:
        ptz = PTZController()
        ptz.connect()
        tracker = AutoTracker(ptz, frame_w=800, frame_h=448)
        tracker.start()
        tracker.update(detections)   # called from main loop
        tracker.stop()
    """

    def __init__(self, ptz: PTZController, frame_w: int, frame_h: int):
        self._ptz     = ptz
        self._frame_w = frame_w
        self._frame_h = frame_h

        self._lock           = threading.Lock()
        self._pending:       Optional[List[Detection]] = None
        self._thread:        Optional[threading.Thread] = None
        self._running        = False
        self._paused         = False

        self._lost_count        = 0
        self._homed             = True   # start in patrol mode immediately
        self._last_zoom_dir     = 0
        self._last_cmd_time     = 0.0
        self._target_stable_since: Optional[float] = None
        self._locked_class_id:  Optional[int] = None
        self._confirm_count     = 0
        self._zoom_out_until    = 0.0  # epoch time to stop zoom-out after loss

        # Patrol state
        _n = len(config.PATROL_PRESETS)
        self._patrol_idx         = (_n - 1) if _n else 0   # wraps to 0 on first step
        self._patrol_dwell_until = 0.0                     # expired → step immediately
        self._detection_preset   = config.PATROL_PRESETS[0] if _n else config.HOME_PRESET

        # Schedule state
        self._scheduled_paused = False

        self.active_target:  Optional[Detection] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop, daemon=True, name="tracker"
        )
        self._thread.start()
        log.info("Tracker started (frame %dx%d)", self._frame_w, self._frame_h)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._ptz.stop()

    def pause(self, state: bool):
        self._paused = state
        if state:
            self._ptz.stop()
            self._ptz.zoom(0)
            log.info("Tracker paused")
        else:
            log.info("Tracker resumed")

    def reset_home(self):
        self._ptz.stop()
        self._ptz.zoom(0)
        self._ptz.goto_preset(config.HOME_PRESET)
        self._homed = True
        self._lost_count = 0
        log.info("Tracker: manual home reset")

    def update(self, detections: Optional[List[Detection]]):
        with self._lock:
            self._pending = detections

    # ── Internal ──────────────────────────────────────────────────────────────

    def _control_loop(self):
        while self._running:
            time.sleep(0.033)   # ~30 Hz

            self._check_schedule()

            if self._paused or self._scheduled_paused:
                continue

            with self._lock:
                detections = self._pending
                self._pending = None

            if detections is None:
                continue

            target = self._select_target(detections)

            # Stop zoom-out once timer expires
            if self._zoom_out_until and time.time() > self._zoom_out_until:
                self._ptz.zoom(0)
                self._last_zoom_dir  = 0
                self._zoom_out_until = 0.0

            if target is None:
                self._target_stable_since = None
                self._locked_class_id     = None
                self._confirm_count       = 0
                self.active_target        = None
                self._handle_lost()
                if self._homed and config.PATROL_ENABLED and config.PATROL_PRESETS:
                    self._patrol_step()
            else:
                self._confirm_count += 1
                if self._confirm_count < config.DETECTION_CONFIRM_FRAMES:
                    continue   # not enough consecutive frames yet

                # Transitioning from patrol → tracking: record where we detected
                if self._homed and config.PATROL_PRESETS:
                    self._detection_preset = config.PATROL_PRESETS[self._patrol_idx]
                    log.info("Target acquired at patrol preset %d", self._detection_preset)

                self._lost_count = 0
                self._homed      = False
                self.active_target = target

                if self._target_stable_since is None:
                    self._target_stable_since = time.time()

                stable = (time.time() - self._target_stable_since) >= config.DETECTION_STABLE_S

                if not stable:
                    continue

                moving = self._center_target(target)
                if not moving:
                    self._control_zoom(target)
                else:
                    self._ptz.zoom(0)
                    self._last_zoom_dir = 0

    def _select_target(self, detections: List[Detection]) -> Optional[Detection]:
        candidates = [d for d in detections if d.class_id in config.TARGET_CLASSES]
        if not candidates:
            self._locked_class_id = None
            return None

        # If locked onto a class, prefer it unless a rival is significantly better
        if self._locked_class_id is not None:
            locked = [d for d in candidates if d.class_id == self._locked_class_id]
            if locked:
                best_locked = max(locked, key=lambda d: d.conf)
                best_any    = max(candidates, key=lambda d: d.conf)
                if best_any.conf - best_locked.conf >= config.LOCK_SWITCH_MARGIN:
                    self._locked_class_id = best_any.class_id
                    return best_any
                return best_locked

        # No lock yet — pick highest confidence and lock onto it
        best = max(candidates, key=lambda d: d.conf)
        self._locked_class_id = best.class_id
        return best

    def _handle_lost(self):
        self.active_target = None

        # Already homed — don't send any further commands that would interrupt
        # the camera's movement to the preset position.
        if self._homed:
            return

        self._lost_count += 1
        self._ptz.stop()

        # Only trigger zoom-out once on first lost frame
        if self._lost_count == 1:
            self._ptz.zoom(-1)
            self._last_zoom_dir  = -1
            self._zoom_out_until = time.time() + config.ZOOM_OUT_ON_LOSS_S

        if self._lost_count >= config.LOST_FRAMES_LIMIT:
            log.info("Target lost — returning to preset %d", self._detection_preset)
            self._ptz.stop()
            self._ptz.zoom(0)
            self._ptz.goto_preset(self._detection_preset)
            self._homed          = True
            self._last_zoom_dir  = 0
            self._lost_count     = 0
            self._zoom_out_until = 0.0
            # Resume patrol from the preset we just returned to
            if self._detection_preset in config.PATROL_PRESETS:
                self._patrol_idx = config.PATROL_PRESETS.index(self._detection_preset)
            self._patrol_dwell_until = time.time() + config.PATROL_DWELL_S

    @staticmethod
    def _in_pause_window() -> bool:
        """Return True if the current time falls within the scheduled pause window."""
        now    = datetime.now().time().replace(second=0, microsecond=0)
        pause  = dtime(*map(int, config.SCHEDULE_PAUSE_TIME.split(":")))
        resume = dtime(*map(int, config.SCHEDULE_RESUME_TIME.split(":")))
        if pause <= resume:
            return pause <= now < resume
        # Crosses midnight (e.g. 23:00 → 06:00)
        return now >= pause or now < resume

    def _check_schedule(self):
        if not config.SCHEDULE_ENABLED:
            return
        should_pause = self._in_pause_window()

        if should_pause and not self._scheduled_paused:
            log.info("Schedule: pausing — going to preset %d", config.SCHEDULE_PAUSE_PRESET)
            self._ptz.stop()
            self._ptz.zoom(0)
            self._ptz.goto_preset(config.SCHEDULE_PAUSE_PRESET)
            self._scheduled_paused = True
            self._homed            = True
            self._lost_count       = 0
            self.active_target     = None

        elif not should_pause and self._scheduled_paused:
            log.info("Schedule: resuming patrol")
            self._scheduled_paused   = False
            self._homed              = True
            self._patrol_dwell_until = 0.0   # advance to first patrol preset immediately

    def _patrol_step(self):
        """Advance to the next patrol preset if the dwell time has elapsed."""
        if time.time() < self._patrol_dwell_until:
            return
        self._patrol_idx = (self._patrol_idx + 1) % len(config.PATROL_PRESETS)
        preset = config.PATROL_PRESETS[self._patrol_idx]
        log.info("Patrol: moving to preset %d", preset)
        self._ptz.goto_preset(preset)
        self._patrol_dwell_until = time.time() + config.PATROL_DWELL_S

    def _center_target(self, target: Detection) -> bool:
        """Returns True if a move command was issued."""
        cx = (target.x1 + target.x2) / 2.0
        cy = (target.y1 + target.y2) / 2.0

        norm_x = (cx - self._frame_w / 2.0) / (self._frame_w / 2.0)
        norm_y = (cy - self._frame_h / 2.0) / (self._frame_h / 2.0)

        pan_speed  = self._offset_to_speed(norm_x, config.DEADZONE_X, config.KP_PAN)
        tilt_speed = self._offset_to_speed(norm_y, config.DEADZONE_Y, config.KP_TILT)

        if pan_speed == 0 and tilt_speed == 0:
            self._ptz.stop()
            self._last_cmd_time = 0.0   # reset cooldown when stopped
            return False

        # Throttle: don't update command until camera has had time to execute last one
        now = time.time()
        if now - self._last_cmd_time < config.PTZ_CMD_COOLDOWN_S:
            return True   # still moving from last command

        log.debug("move pan=%d tilt=%d (norm_x=%.2f norm_y=%.2f)",
                  pan_speed, -tilt_speed, norm_x, norm_y)
        self._ptz.move(pan_speed=pan_speed, tilt_speed=-tilt_speed)
        self._last_cmd_time = now
        return True

    def _control_zoom(self, target: Detection):
        # Force zoom out if target is near the frame edge (fast-moving target)
        cx = (target.x1 + target.x2) / 2.0
        cy = (target.y1 + target.y2) / 2.0
        norm_x = abs((cx - self._frame_w / 2.0) / (self._frame_w / 2.0))
        norm_y = abs((cy - self._frame_h / 2.0) / (self._frame_h / 2.0))
        near_edge = norm_x > config.ZOOM_OUT_EDGE_THRESH or norm_y > config.ZOOM_OUT_EDGE_THRESH

        if near_edge:
            zoom_dir = -1   # zoom out to regain FOV
        else:
            bbox_area  = (target.x2 - target.x1) * (target.y2 - target.y1)
            frame_area = self._frame_w * self._frame_h
            ratio      = bbox_area / frame_area if frame_area > 0 else 0

            if ratio < config.ZOOM_IN_THRESH:
                zoom_dir = 1
            elif ratio > config.ZOOM_OUT_THRESH:
                zoom_dir = -1
            else:
                zoom_dir = 0

        if zoom_dir != self._last_zoom_dir:
            self._ptz.zoom(zoom_dir)
            self._last_zoom_dir = zoom_dir

    @staticmethod
    def _offset_to_speed(offset: float, deadzone: float, gain: float) -> int:
        if abs(offset) < deadzone:
            return 0
        effective = (abs(offset) - deadzone) / (1.0 - deadzone)
        speed = int(round(effective * gain))
        speed = max(config.PTZ_PAN_SPEED_MIN, min(speed, config.PTZ_PAN_SPEED_MAX))
        return speed if offset > 0 else -speed
