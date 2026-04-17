"""
PTZ diagnostic — tests XMEye login and basic movement commands.
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from ptz import PTZController

ptz = PTZController()
print("Connecting...")
if not ptz.connect():
    print("FAILED to connect")
    raise SystemExit(1)

print("Connected. Testing pan right for 2s...")
ptz.move(pan_speed=3, tilt_speed=0)
time.sleep(2)
ptz.stop()
print("Stop.")

time.sleep(1)

print("Testing pan left for 2s...")
ptz.move(pan_speed=-3, tilt_speed=0)
time.sleep(2)
ptz.stop()
print("Stop.")

time.sleep(1)

print("Testing tilt up for 2s...")
ptz.move(pan_speed=0, tilt_speed=3)
time.sleep(2)
ptz.stop()
print("Stop.")

time.sleep(1)

print("Testing zoom in for 2s...")
ptz.zoom(1)
time.sleep(2)
ptz.zoom(0)
print("Stop zoom.")

time.sleep(1)

print("Returning to preset 1...")
ptz.goto_preset(1)
time.sleep(2)

ptz.disconnect()
print("Done.")
