"""
XMEye/Sofia PTZ controller for Sunba 405-D20X ECO.

Protocol: binary 20-byte header + JSON body over TCP port 34567.
MSG_PTZ = 0x0578 (OPPTZControl) confirmed for this camera.

Movement requires continuous command sending at _MOVE_HZ; a heartbeat
thread handles this automatically after move() is called.
"""

import hashlib
import json
import logging
import random
import socket
import struct
import threading
import time
from typing import Optional

import config

log = logging.getLogger(__name__)

# ── Protocol constants ────────────────────────────────────────────────────────
# 20-byte header: magic+pad(4) | session_id(4) | seq(4) | channel(1) | ack(1) | code(2) | len(4)
PACKET_FMT  = "<4sIIBBHI"
PACKET_SIZE = struct.calcsize(PACKET_FMT)   # 20 bytes
MAGIC       = b"\xff\x01\x00\x00"

MSG_LOGIN   = 0x03E8   # 1000
MSG_PTZ     = 0x0578   # 1400 — OPPTZControl confirmed for Sunba 405-D20X ECO
MSG_KEEPALIVE = 0x03EE # 1006

_MOVE_HZ    = 0.2      # seconds between heartbeat PTZ commands while moving


def _md5_password(password: str) -> str:
    """XMEye-style truncated MD5 password hash."""
    raw = hashlib.md5(password.encode()).digest()
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result = ""
    for i in range(0, 16, 2):
        val = (raw[i] + raw[i + 1]) % len(chars)
        result += chars[val]
    return result


def _random_salt() -> str:
    return str(random.randint(100000, 999999))


class PTZController:
    """
    XMEye/Sofia PTZ controller.

    Usage:
        ptz = PTZController()
        ptz.connect()
        ptz.move(pan_speed=3, tilt_speed=0)   # pan right
        ptz.stop()
        ptz.goto_preset(1)
        ptz.disconnect()
    """

    def __init__(self):
        self._sock:       Optional[socket.socket] = None
        self._session_id: int  = 0
        self._seq:        int  = 0
        self._lock        = threading.Lock()
        self._connected   = False

        # Heartbeat state
        self._hb_thread:  Optional[threading.Thread] = None
        self._hb_running  = False
        self._move_cmd:   Optional[str] = None   # e.g. "Right", "Up"
        self._move_speed: int = 0
        self._zoom_cmd:   Optional[str] = None   # "ZoomTile" / "ZoomWide"

        # Keepalive
        self._ka_thread:  Optional[threading.Thread] = None
        self._ka_running  = False

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5)
            self._sock.connect((config.CAMERA_IP, config.PTZ_PORT))
        except OSError as exc:
            log.error("PTZ connect failed: %s", exc)
            return False

        if not self._login():
            return False

        self._connected = True
        self._start_heartbeat()
        self._start_keepalive()
        log.info("PTZ connected to %s:%d", config.CAMERA_IP, config.PTZ_PORT)
        return True

    def disconnect(self):
        self._connected = False
        self._stop_heartbeat()
        self._stop_keepalive()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def move(self, pan_speed: int = 0, tilt_speed: int = 0):
        """
        Start continuous movement.
        pan_speed  : negative=left, positive=right (0 = no pan)
        tilt_speed : negative=down, positive=up    (0 = no tilt)
        """
        cmd, speed = self._resolve_direction(pan_speed, tilt_speed)
        with self._lock:
            self._move_cmd   = cmd
            self._move_speed = speed
        self._ptz_command(cmd, speed, preset=65535)   # send immediately; heartbeat sustains

    def stop(self):
        """Stop all movement."""
        with self._lock:
            self._move_cmd   = None
            self._move_speed = 0
        self._ptz_command("DirectionLeft", 0, preset=-1)   # release

    def zoom(self, direction: int):
        """direction: 1=zoom in, -1=zoom out, 0=stop zoom."""
        with self._lock:
            if direction > 0:
                self._zoom_cmd = "ZoomTile"
            elif direction < 0:
                self._zoom_cmd = "ZoomWide"
            else:
                self._zoom_cmd = None

    def goto_preset(self, preset_num: int):
        """Send camera to a saved preset position."""
        payload = {
            "Name": "OPPTZControl",
            "SessionID": f"0x{self._session_id:08X}",
            "OPPTZControl": {
                "Command":   "GotoPreset",
                "Parameter": {
                    "AUX":      {"Number": 0, "Status": "On"},
                    "Channel":  config.PTZ_CHANNEL,
                    "MenuOpts": "Enter",
                    "POINT":    {"bottom": 0, "left": 0, "right": 0, "top": 0},
                    "Pattern":  "SetBegin",
                    "Preset":   preset_num,
                    "Step":     0,
                    "Tour":     0,
                },
            },
        }
        self._send(MSG_PTZ, payload)

    # ── Internal: connection ──────────────────────────────────────────────────

    def _login(self) -> bool:
        payload = {
            "EncryptType": "MD5",
            "LoginType":   "DVRIP-Web",
            "PassWord":    _md5_password(config.CAMERA_PASSWORD),
            "UserName":    config.CAMERA_USER,
        }
        self._send(MSG_LOGIN, payload)
        resp = self._recv()
        if resp is None:
            log.error("PTZ login: no response")
            return False
        ret = resp.get("Ret", -1)
        if ret not in (100, 101):
            log.error("PTZ login failed: Ret=%s", ret)
            return False
        sid = resp.get("SessionID", "0x0")
        self._session_id = int(sid, 16) if isinstance(sid, str) else int(sid)
        log.info("PTZ login OK, session=0x%08X", self._session_id)
        return True

    # ── Internal: heartbeat ───────────────────────────────────────────────────

    def _start_heartbeat(self):
        self._hb_running = True
        self._hb_thread  = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="ptz-heartbeat"
        )
        self._hb_thread.start()

    def _stop_heartbeat(self):
        self._hb_running = False
        if self._hb_thread:
            self._hb_thread.join(timeout=2)

    def _heartbeat_loop(self):
        while self._hb_running:
            time.sleep(_MOVE_HZ)
            with self._lock:
                move_cmd   = self._move_cmd
                move_speed = self._move_speed
                zoom_cmd   = self._zoom_cmd

            if move_cmd:
                self._ptz_command(move_cmd, move_speed, preset=65535)
            if zoom_cmd:
                self._ptz_command(zoom_cmd, config.PTZ_ZOOM_SPEED, preset=65535)

    # ── Internal: keepalive ───────────────────────────────────────────────────

    def _start_keepalive(self):
        self._ka_running = True
        self._ka_thread  = threading.Thread(
            target=self._keepalive_loop, daemon=True, name="ptz-keepalive"
        )
        self._ka_thread.start()

    def _stop_keepalive(self):
        self._ka_running = False
        if self._ka_thread:
            self._ka_thread.join(timeout=2)

    def _keepalive_loop(self):
        while self._ka_running:
            time.sleep(config.PTZ_KEEPALIVE_S)
            payload = {"Name": "KeepAlive", "SessionID": f"0x{self._session_id:08X}"}
            self._send(MSG_KEEPALIVE, payload)

    # ── Internal: PTZ command ─────────────────────────────────────────────────

    def _ptz_command(self, command: str, speed: int, preset: int = 65535):
        payload = {
            "Name":      "OPPTZControl",
            "SessionID": f"0x{self._session_id:08X}",
            "Salt":      _random_salt(),
            "OPPTZControl": {
                "Command": command,
                "Parameter": {
                    "AUX":      {"Number": 0, "Status": "On"},
                    "Channel":  config.PTZ_CHANNEL,
                    "MenuOpts": "Enter",
                    "POINT":    {"bottom": 0, "left": 0, "right": 0, "top": 0},
                    "Pattern":  "SetBegin",
                    "Preset":   preset,
                    "Step":     speed,
                    "Tour":     0,
                },
            },
        }
        self._send(MSG_PTZ, payload)

    @staticmethod
    def _resolve_direction(pan: int, tilt: int):
        """Map signed pan/tilt speeds to an XMEye direction command + speed."""
        if pan != 0 and tilt != 0:
            if pan > 0 and tilt > 0:
                return "DirectionRightUp",   max(abs(pan), abs(tilt))
            if pan > 0 and tilt < 0:
                return "DirectionRightDown", max(abs(pan), abs(tilt))
            if pan < 0 and tilt > 0:
                return "DirectionLeftUp",    max(abs(pan), abs(tilt))
            return "DirectionLeftDown", max(abs(pan), abs(tilt))
        if pan > 0:
            return "DirectionRight", abs(pan)
        if pan < 0:
            return "DirectionLeft",  abs(pan)
        if tilt > 0:
            return "DirectionUp",    abs(tilt)
        if tilt < 0:
            return "DirectionDown",  abs(tilt)
        return "DirectionLeft", 0

    # ── Internal: transport ───────────────────────────────────────────────────

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload, separators=(",", ":")).encode() + b"\x0a"
        header = struct.pack(
            PACKET_FMT,
            MAGIC,
            self._session_id,
            self._seq,
            0, 0,
            code,
            len(body),
        )
        self._seq += 1
        try:
            with self._lock:
                self._sock.sendall(header + body)
        except OSError as exc:
            log.warning("PTZ send error: %s", exc)

    def _recv(self) -> Optional[dict]:
        try:
            header = self._sock.recv(PACKET_SIZE)
            if len(header) < PACKET_SIZE:
                return None
            _, _, _, _, _, code, length = struct.unpack(PACKET_FMT, header)
            body = b""
            while len(body) < length:
                chunk = self._sock.recv(length - len(body))
                if not chunk:
                    break
                body += chunk
            text = body.rstrip(b"\x0a").decode(errors="ignore")
            obj, _ = json.JSONDecoder().raw_decode(text)
            return obj
        except Exception as exc:
            log.warning("PTZ recv error: %s", exc)
            return None
