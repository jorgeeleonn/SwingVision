import os

# All the parameters of the project are here, so nothing has to be changed inside the code

# ---------------------------------------------------------------- Paths (relative to src)
DATA_PATH = os.path.join("..", "data")
RESULTS_PATH = os.path.join("..", "results")
CALIBRATION_FILE = os.path.join(RESULTS_PATH, "calibration", "calibration.npz")
REFERENCE_PATH = os.path.join(RESULTS_PATH, "reference")

# ---------------------------------------------------------------- Camera and video
CAMERA_INDEX = 0             # camera used in live mode (same meaning as in test.py)
CAMERA_SIZE = (1280, 720)    # resolution asked to the camera
SWING_VIDEO = None       # path of a recorded swing to analyse it instead of using the live camera
SECURITY_VIDEO = None        # path of a video of the cards instead of the live camera (only for tests)
RECORD_FILE = None           # path of an .avi file to save the whole session (for the demo)
VIDEO_FPS = 240              # frames per second of the recorded swings (None = read it from the file)
WORK_WIDTH = 960             # every frame is resized to this width before processing (None = keep the size)
BACKGROUND_SECONDS = 0.5     # seconds of empty scene used to build the background
MAX_FRAMES = 3000            # frames kept in memory by the tracker (the oldest ones are forgotten)

# ---------------------------------------------------------------- Calibration
PATTERN_SIZE = (8, 6)        # inner corners of the chessboard (columns, rows)
SQUARE_SIZE = 30             # side of a square [mm]
MM_PER_PX = None             # write a value here to skip the chessboard in the plane of the player

# ---------------------------------------------------------------- Security system
# Code of every player: the colours of the four flags, in order
USERS = {
    "jorge": ["red", "blue", "green", "black"],
    "alex": ["black", "green", "red", "blue"],
}
# Hue ranges (OpenCV: 0-179) of the coloured pennants. Red is at both ends of the hue circle
FLAG_HUES = {
    "red": [(0, 4), (160, 179)],
    "green": [(40, 90)],
    "blue": [(95, 140)],
}
COLOR_MIN_SATURATION = 90    # coloured ink is more saturated than this (S channel)
COLOR_MIN_VALUE = 50         # and brighter than this (V channel)
BLACK_MAX_VALUE = 85         # black ink is darker than this...
BLACK_MAX_SATURATION = 110   # ...and less saturated than this (a dark blue pennant is dark, but saturated)
PENNANT_AREA = (0.0004, 0.02)  # minimum and maximum area of the pennant (fraction of the image)
MIN_TRIANGULARITY = 0.70     # area of the pennant / area of the smallest triangle around it
PAPER_MAX_SATURATION = 75    # the paper around the pennant has less colour than this
PAPER_MIN_VALUE = 105        # and is brighter than this
MIN_PAPER_AROUND = 0.65      # fraction of the pixels around the pennant that must look like paper
POLE_BLOCK_SIZE = 21         # neighbourhood of the adaptive threshold that finds the ink of the pole [px]
POLE_OFFSET = 10             # the ink is this much darker than the mean of its neighbourhood
MIN_POLE_LENGTH = 1.1        # minimum length of the pole / size of the pennant
MIN_PAPER_BESIDE_POLE = 0.5  # fraction of paper on each side of the pole
ACCEPT_FRAMES = 8            # frames with the same flag needed to accept it
RELEASE_FRAMES = 5           # consecutive frames without any flag needed before accepting the next one
SEQUENCE_TIMEOUT = 10        # seconds without showing a flag before the sequence is forgotten

# ---------------------------------------------------------------- Player segmentation
DIFF_THRESHOLD = 30          # minimum difference with the background to be foreground (0-255)
OPEN_SIZE = 9                # kernel of the opening: larger than the shaft, so the club is not part of the silhouette [px]
CLOSE_SIZE = 15              # kernel of the closing that fills holes [px]
MIN_PLAYER_AREA = 0.01       # minimum area of the silhouette (fraction of the image)

# ---------------------------------------------------------------- Tracker
# HSV range of every marker: (lower, upper). Use utils.get_hsv_color_ranges() to find them
MARKERS = {
    "cap": ((140, 100, 100), (170, 255, 255)),     # magenta tape on the cap
    "grip": ((20, 120, 120), (35, 255, 255)),      # yellow tape on the shaft, just below the hands
    "club": ((85, 120, 100), (100, 255, 255)),     # cyan tape on the club head
}
MARKER_COLORS = {"cap": (255, 0, 255), "grip": (0, 255, 255), "club": (255, 255, 0)}  # BGR, only to draw
MIN_MARKER_AREA = 10         # minimum area of a marker [px]
SEARCH_RADIUS = 100          # maximum distance between the prediction and the detection [px]
MAX_LOST_FRAMES = 20         # frames without detection before the marker is searched in the whole image
KALMAN_PROCESS_NOISE = 1.0   # larger = the filter follows fast changes of velocity better
TRAIL_LENGTH = 60            # points of the trajectories drawn on the video

# Club shaft (Canny + Hough)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_MAX_GAP = 10           # maximum gap between points of the same segment [px]
HOUGH_MAX_ANGLE = 12         # maximum difference between the Hough line and the markers [degrees]

# ---------------------------------------------------------------- Detection of the P positions
SMOOTH_SECONDS = 0.025       # window of the moving average applied to the trajectories [s]
STILL_SPEED_RATIO = 0.02     # the hands are still below this fraction of their maximum speed
MAX_BACKSWING_SECONDS = 2.0  # the top of the swing is searched this long before the fastest moment
MAX_DOWNSWING_SECONDS = 0.8  # the impact is searched this long after the top of the swing
POSITIONS = ["P1", "P2", "P4", "P6", "P7"]

# ---------------------------------------------------------------- Comparison with the reference
K_VERDE = 2.0                # difference below K_VERDE * sigma -> "bien" (green)
K_ROJO = 4.0                 # difference above K_ROJO * sigma -> "mal" (red). In between -> "a mejorar"

# Minimum green tolerance of every kind of measure, so that the noise is never marked as an error
TOL_MIN = {
    "position": 0.03,        # fraction of the height of the player
    "angle": 5.0,            # degrees
    "tempo": 0.3,            # backswing / downswing ratio
    "hands_speed": 1.0,      # m/s
    "club_speed": 3.0,       # m/s
    "time": 10.0,            # ms
}

# Measures compared in every position (all of them are computed, these are the ones shown)
MEASURES = {
    "P1": ["hands_x", "hands_y", "tilt"],
    "P2": ["hands_x", "hands_y", "head_x", "head_y", "sway", "tilt"],
    "P4": ["hands_x", "hands_y", "head_x", "head_y", "club", "sway", "tilt"],
    "P6": ["hands_x", "hands_y", "head_x", "head_y", "sway", "tilt"],
    "P7": ["hands_x", "hands_y", "head_x", "head_y", "club", "sway", "tilt"],
}
GLOBAL_MEASURES = ["tempo", "hands_speed", "club_speed", "club_peak"]
