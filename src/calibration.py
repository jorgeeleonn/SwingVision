import os
import cv2
import numpy as np
from typing import List

import config
from utils import get_chessboard_points


def find_corners(imgs: List, pattern_size: tuple) -> List:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = [cv2.findChessboardCorners(img, pattern_size, None) for img in imgs]
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    # cornerSubPix needs grayscale images and works in place
    corners_refined = [cv2.cornerSubPix(gray, cor[1], (5, 5), (-1, -1), criteria) if cor[0] else None
                       for gray, cor in zip(imgs_gray, corners)]
    return [(cor[0], cor_ref) for cor, cor_ref in zip(corners, corners_refined)]


def calibrate_camera(imgs: List, pattern_size: tuple, square_size: float):
    corners = find_corners(imgs, pattern_size)
    valid_corners = np.array([cor[1] for cor in corners if cor[0]], dtype=np.float32)
    chessboard_points = [get_chessboard_points(pattern_size, square_size, square_size) for cor in corners if cor[0]]

    image_size = imgs[0].shape[1::-1]
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(chessboard_points, valid_corners, image_size, None, None)
    return rms, intrinsics, dist_coeffs, corners


def compute_scale(img: np.array, pattern_size: tuple, square_size: float):
    """The board must be where the player stands, facing the camera. Returns None if it is not found"""
    # The board is small at that distance, so the image is enlarged if it is not found at its size
    for zoom in [1, 2, 3]:
        zoomed = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        found, corners = find_corners([zoomed], pattern_size)[0]
        if found:
            break
    if not found:
        return None
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2) / zoom
    # Mean distance between neighbouring corners, along the rows and along the columns
    step_rows = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
    step_cols = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
    return square_size / np.mean([step_rows, step_cols])


def save_calibration(path: str, rms: float, intrinsics: np.array, dist_coeffs: np.array, image_size: tuple, mm_per_px: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, rms=rms, intrinsics=intrinsics, dist_coeffs=dist_coeffs, image_size=image_size,
             mm_per_px=np.nan if mm_per_px is None else mm_per_px)


def load_calibration(path: str = config.CALIBRATION_FILE):
    if not os.path.exists(path):
        print("Warning: there is no calibration file, the frames will not be undistorted")
        return None
    data = np.load(path)
    calib = {key: data[key] for key in data.files}
    calib["image_size"] = tuple(int(v) for v in calib["image_size"])
    calib["mm_per_px"] = None if np.isnan(calib["mm_per_px"]) else float(calib["mm_per_px"])

    # For every pixel of the corrected image, map1/map2 say where to read it in the distorted image.
    # The corrected image is built directly at the working size, so correcting and resizing cost a single remap
    width, height = calib["image_size"]
    scale = config.WORK_WIDTH / width if config.WORK_WIDTH is not None else 1
    new_intrinsics = calib["intrinsics"].copy()
    new_intrinsics[:2] *= scale
    new_size = (int(round(width * scale)), int(round(height * scale)))
    calib["map1"], calib["map2"] = cv2.initUndistortRectifyMap(calib["intrinsics"], calib["dist_coeffs"], np.eye(3),
                                                               new_intrinsics, new_size, cv2.CV_16SC2)
    return calib


def undistort(img: np.array, calib: dict) -> np.array:
    return cv2.undistort(img, calib["intrinsics"], calib["dist_coeffs"])


def prepare_frame(frame: np.array, calib: dict) -> np.array:
    """Every frame goes through this function before being processed: distortion correction and resize"""
    if calib is not None and frame.shape[1::-1] == calib["image_size"]:
        return cv2.remap(frame, calib["map1"], calib["map2"], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Without calibration (or with another resolution) the frame is only resized
    if calib is not None and not calib.get("warned", False):
        print(f"Warning: image size {frame.shape[1::-1]} is not the calibrated size {calib['image_size']}, distortion not corrected")
        calib["warned"] = True
    if config.WORK_WIDTH is not None and frame.shape[1] != config.WORK_WIDTH:
        scale = config.WORK_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def get_scale(calib: dict, frame_width: int):
    if config.MM_PER_PX is not None:
        return config.MM_PER_PX
    if calib is None or calib["mm_per_px"] is None:
        return None
    # mm_per_px was measured at the calibrated size, the frame may have been resized afterwards
    return calib["mm_per_px"] * calib["image_size"][0] / frame_width
