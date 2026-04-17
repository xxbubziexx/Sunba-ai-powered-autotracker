"""
RTSP stream capture thread using ffpyplayer.

Puts the latest (frame, timestamp) into a bounded queue.
Old frames are dropped to stay real-time.
"""

import logging
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np

import config

log = logging.getLogger(__name__)


class StreamCapture:
    """
    Captures RTSP frames in a background thread.

    Usage:
        sc = StreamCapture()
        sc.start()
        result = sc.get_frame(timeout=0.1)
        if result:
            frame, ts = result
        sc.stop()
    """

    def __init__(self):
        self._q:        queue.Queue = queue.Queue(maxsize=config.STREAM_QUEUE_SIZE)
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self.connected  = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(
            target=self._capture_loop, daemon=True, name="stream"
        )
        self._thread.start()
        log.info("Stream capture thread started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Return (bgr_frame, timestamp) or None if queue is empty."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self):
        from ffpyplayer.player import MediaPlayer

        while self._running:
            self.connected = False
            try:
                player = MediaPlayer(
                    config.RTSP_URL,
                    ff_opts=config.FF_OPTS,
                    lib_opts=config.LIB_OPTS,
                )
                log.info("Stream opened: %s", config.RTSP_URL)
                self.connected = True

                while self._running:
                    frame, val = player.get_frame()
                    if val == "eof":
                        log.warning("Stream EOF — reconnecting")
                        break
                    if frame is None:
                        time.sleep(0.005)
                        continue

                    img, pts = frame
                    w, h = img.get_size()
                    arr = np.frombuffer(img.to_bytearray()[0], dtype=np.uint8)
                    arr = arr.reshape((h, w, 3))

                    import cv2
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                    # Drop oldest frame if queue full
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        self._q.put_nowait((bgr, pts))
                    except queue.Full:
                        pass

                player.close_player()

            except Exception as exc:
                log.error("Stream error: %s — retrying in 3s", exc)
                self.connected = False
                time.sleep(3)
