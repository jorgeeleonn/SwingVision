import os
import glob

import config
from calibration import load_calibration
from comparison import register_reference

# Offline step, like the calibration: it is run once, before using the system.
# It analyses the swings of the reference player and saves the reference used in the comparison

if __name__ == "__main__":
    calib = load_calibration()
    videopaths = sorted(glob.glob(os.path.join(config.DATA_PATH, "videos", "reference", "*.mp4")))
    print(f"{len(videopaths)} reference videos found")
    if videopaths:
        register_reference(videopaths, calib)
