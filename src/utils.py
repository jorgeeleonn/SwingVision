import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def show_images(images: List, titles: List, bgr: bool = True, size: int = 5):
    # Displays several images side by side inside the notebook
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size * 0.6))
    axes = np.atleast_1d(axes)
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:  # grayscale or binary image
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]


def write_image(output_folder: str, img_name: str, img: np.array):
    os.makedirs(output_folder, exist_ok=True)
    img_path = os.path.join(output_folder, img_name)
    cv2.imwrite(img_path, img)


def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.
    Only for short videos: a swing at 240 fps does not fit in memory, so it is read frame by frame in swing.py

    Returns:
        tuple: frames (list), frame_width (int), frame_height (int), frame_rate (float)
    """
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print('Error: Could not open the video file')
        return [], 0, 0, 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate


def extract_frames(videopath: str, output_folder: str, step: int = 30):
    """Saves one frame out of every 'step' frames of a video (used to get the calibration images
    from a video recorded in the same mode as the swings)"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(videopath)
    saved = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved:03d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"{saved} frames saved in {output_folder}")


def get_chessboard_points(chessboard_shape, dx, dy):
    points_per_row, points_per_col = chessboard_shape
    points = []
    # Same order as cv2.findChessboardCorners()
    for row in range(points_per_col):
        for col in range(points_per_row):
            # The X axis must follow the short side of the pattern and Y the long side
            if points_per_row >= points_per_col:
                points.append([row * dx, col * dy, 0])
            else:
                points.append([col * dx, row * dy, 0])
    return np.array(points, dtype=np.float32)


def draw_lines(img, lines, color, thickness):
    """Draw every line returned by cv2.HoughLinesP() on img (in place) and return it"""
    if lines is None:  # Hough did not find any line
        return img
    # Returns reshape to (N, 4)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def draw_text(img, text, position, color=(255, 255, 255), scale=0.6, thickness=2):
    """Writes text with a black border so that it can be read over any background"""
    # The fonts of OpenCV have no accents
    text = text.translate(str.maketrans("áéíóúñ", "aeioun"))
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img


def nothing(x):
    pass


def get_hsv_color_ranges(image: np.array):
    # Same tool as in Lab 1: move the trackbars until only the marker is visible and copy the printed range

    # Create a window
    cv2.namedWindow('image')

    # Create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 255)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break
        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        cv2.imshow('image',output)
        cv2.resizeWindow("image", 500,300)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
