import time
import cv2
import numpy as np
from typing import List

import config
from calibration import prepare_frame, get_scale
from utils import draw_text


def build_background(frames: List) -> np.array:
    # The median of several frames of the empty scene removes the noise
    step = max(1, len(frames) // 15)
    background = np.median(np.stack(frames[::step]), axis=0).astype(np.uint8)
    return cv2.GaussianBlur(background, (5, 5), 0)


def segment_player(frame: np.array, background: np.array):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # Difference with the empty scene: the largest of the three channels, so a colour change is enough
    b, g, r = cv2.split(cv2.absdiff(blurred, background))
    diff = cv2.max(cv2.max(b, g), r)
    _, foreground = cv2.threshold(diff, config.DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Opening removes small blobs and the club (it is thinner than the kernel), closing fills the holes inside the body
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.OPEN_SIZE, config.OPEN_SIZE))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.CLOSE_SIZE, config.CLOSE_SIZE))
    mask = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    # The player is the largest region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return foreground, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < config.MIN_PLAYER_AREA * mask.size:
        return foreground, None
    silhouette = np.zeros_like(mask)
    cv2.drawContours(silhouette, [largest], -1, 255, -1)
    return foreground, silhouette


def get_silhouette_pose(silhouette: np.array):
    # Centre of the silhouette and tilt of its main axis from the vertical [degrees, positive = leaning right]
    m = cv2.moments(silhouette, binaryImage=True)
    centre = (m["m10"] / m["m00"], m["m01"] / m["m00"])
    # Orientation of the main axis from the central moments
    angle = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))
    tilt = angle - 90 if angle > 0 else angle + 90
    return centre, tilt


class MarkerTracker:

    def __init__(self, name: str):
        self.name = name
        self.lower, self.upper = config.MARKERS[name]
        # Same filter as in Lab 5: state (x, y, vx, vy), measurement (x, y), constant velocity
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
        self.reset()

    def reset(self):
        self.initialised = False
        self.lost = 0            # consecutive frames without detection
        self.position = None     # (x, y) of the marker: measured, or predicted when it is not seen
        self.box = None          # bounding box (x, y, w, h) of the detection
        self.detected = False

    def update(self, hsv: np.array, search: np.array):
        mask = cv2.bitwise_and(cv2.inRange(hsv, self.lower, self.upper), search)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) >= config.MIN_MARKER_AREA]

        best = None
        if not self.initialised:
            # First time: the largest blob of that colour
            if contours:
                best = max(contours, key=cv2.contourArea)
        else:
            # Afterwards: the blob closest to the predicted position, if it is close enough
            prediction = self.kf.predict()[:2].ravel()
            best_distance = config.SEARCH_RADIUS
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                distance = np.hypot(x + w / 2 - prediction[0], y + h / 2 - prediction[1])
                if distance < best_distance:
                    best, best_distance = contour, distance

        if best is not None:
            m = cv2.moments(best)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            if not self.initialised:
                self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
                self.initialised = True
            else:
                self.kf.correct(np.array([[cx], [cy]], np.float32))
            self.position, self.box, self.detected, self.lost = (cx, cy), cv2.boundingRect(best), True, 0
        elif self.initialised:
            # Not seen (hidden or blurred): keep the prediction for a while
            self.position, self.box, self.detected = (float(prediction[0]), float(prediction[1])), None, False
            self.lost += 1
            if self.lost > config.MAX_LOST_FRAMES:
                self.reset()
        return self.position


def find_club_line(gray: np.array, search: np.array, grip: tuple, club: tuple):
    length = np.hypot(club[0] - grip[0], club[1] - grip[1])
    if length < 20:
        return None, None
    # Angle given by the markers (the y axis of the image points down)
    marker_angle = np.degrees(np.arctan2(grip[1] - club[1], club[0] - grip[0]))

    # Region of interest around both markers
    x0, x1 = int(max(min(grip[0], club[0]) - 15, 0)), int(max(grip[0], club[0]) + 15)
    y0, y1 = int(max(min(grip[1], club[1]) - 15, 0)), int(max(grip[1], club[1]) + 15)
    edges = cv2.Canny(gray[y0:y1, x0:x1], config.CANNY_LOW, config.CANNY_HIGH)
    edges = cv2.bitwise_and(edges, search[y0:y1, x0:x1])
    threshold = max(15, int(0.3 * length))   # minimum number of votes
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=0.4 * length, maxLineGap=config.HOUGH_MAX_GAP)
    if lines is None:
        return None, None

    # The shaft is the longest segment that agrees with the markers (the legs and the arms also give lines)
    best_angle, best_segment, best_length = None, None, 0
    for xa, ya, xb, yb in lines.reshape(-1, 4) + np.array([x0, y0, x0, y0]):
        angle = np.degrees(np.arctan2(ya - yb, xb - xa))
        difference = (angle - marker_angle + 90) % 180 - 90   # a segment has no direction: difference in [-90, 90)
        middle = ((xa + xb) / 2, (ya + yb) / 2)
        distance = abs((club[0] - grip[0]) * (grip[1] - middle[1]) - (grip[0] - middle[0]) * (club[1] - grip[1])) / length
        segment_length = np.hypot(xb - xa, yb - ya)
        if abs(difference) < config.HOUGH_MAX_ANGLE and distance < 8 and segment_length > best_length:
            best_angle, best_segment, best_length = marker_angle + difference, (xa, ya, xb, yb), segment_length
    return best_angle, best_segment


class SwingTracker:
    KEYS = ["t", "cap", "grip", "club", "detected", "angle", "centre", "tilt", "jpg"]

    def __init__(self, background: np.array, mm_per_px: float = None):
        self.background = background
        self.mm_per_px = mm_per_px
        self.markers = {name: MarkerTracker(name) for name in config.MARKERS}
        self.trails = {name: [] for name in ["grip", "club"]}
        self.clear()

    def clear(self):
        self.data = {key: [] for key in self.KEYS}

    def update(self, frame: np.array, t: float):
        foreground, silhouette = segment_player(frame, self.background)
        # The markers and the shaft are only searched where something has changed with respect to the background
        search = cv2.dilate(foreground, np.ones((5, 5), np.uint8))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        positions = {name: marker.update(hsv, search) for name, marker in self.markers.items()}

        angle, segment = None, None
        if positions["grip"] is not None and positions["club"] is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            angle, segment = find_club_line(gray, search, positions["grip"], positions["club"])
            if angle is None:  # Hough did not find the shaft: angle of the markers
                grip, club = positions["grip"], positions["club"]
                angle = np.degrees(np.arctan2(grip[1] - club[1], club[0] - grip[0]))
        centre, tilt = get_silhouette_pose(silhouette) if silhouette is not None else ((np.nan, np.nan), np.nan)

        nan = (np.nan, np.nan)
        self.data["t"].append(t)
        for name in self.markers:
            self.data[name].append(positions[name] if positions[name] is not None else nan)
        self.data["detected"].append([self.markers[name].detected for name in self.markers])
        self.data["angle"].append(angle if angle is not None else np.nan)
        self.data["centre"].append(centre)
        self.data["tilt"].append(tilt)
        # The frame is kept compressed, to show the P positions when the swing ends
        self.data["jpg"].append(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
        if len(self.data["t"]) > config.MAX_FRAMES:   # the oldest frame is forgotten
            for key in self.KEYS:
                self.data[key].pop(0)
        for name in self.trails:
            if self.markers[name].detected:
                self.trails[name] = (self.trails[name] + [positions[name]])[-config.TRAIL_LENGTH:]
        self.silhouette, self.segment = silhouette, segment

    def draw(self, frame: np.array) -> np.array:
        if self.silhouette is not None:
            contours, _ = cv2.findContours(self.silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            tinted = frame.copy()
            cv2.drawContours(tinted, contours, -1, (0, 200, 0), -1)
            cv2.addWeighted(tinted, 0.25, frame, 0.75, 0, dst=frame)
            cv2.drawContours(frame, contours, -1, (0, 200, 0), 1)
        for name, trail in self.trails.items():
            if len(trail) > 1:
                cv2.polylines(frame, [np.int32(trail)], False, config.MARKER_COLORS[name], 1, cv2.LINE_AA)
        if self.segment is not None:
            xa, ya, xb, yb = [int(v) for v in self.segment]
            cv2.line(frame, (xa, ya), (xb, yb), (0, 0, 255), 2)
        for name, marker in self.markers.items():
            if marker.box is not None:
                x, y, w, h = marker.box
                cv2.rectangle(frame, (x - 4, y - 4), (x + w + 4, y + h + 4), config.MARKER_COLORS[name], 2)
                draw_text(frame, name, (x + w + 8, y + h), config.MARKER_COLORS[name], 0.5, 1)
        return frame

    def get_swing(self) -> dict:
        swing = {key: np.array(self.data[key], dtype=float) for key in self.KEYS if key != "jpg"}
        swing["jpg"] = list(self.data["jpg"])
        swing["background"] = self.background
        swing["mm_per_px"] = self.mm_per_px
        t = swing["t"]
        swing["fps"] = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else 0
        return swing


def analyse_video(videopath: str, calib: dict, show: bool = False, save_path: str = None):
    """The video must begin with the empty scene. save_path - saves the tracking as a video"""
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print('Error: Could not open the video file')
        return None
    fps = config.VIDEO_FPS if config.VIDEO_FPS is not None else cap.get(cv2.CAP_PROP_FPS)

    n_background = int(config.BACKGROUND_SECONDS * fps)
    frames = []
    for _ in range(n_background):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(prepare_frame(frame, calib))
    background = build_background(frames)
    tracker = SwingTracker(background, get_scale(calib, background.shape[1]))

    out = None
    if save_path is not None:
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, background.shape[1::-1])
    i = n_background
    previous = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = prepare_frame(frame, calib)
        tracker.update(frame, i / fps)
        i += 1
        if show or out is not None:
            now = time.time()
            tracker.draw(frame)
            draw_text(frame, f"FPS: {1 / max(now - previous, 1e-6):.0f}", (10, 30), (0, 255, 0), 0.8)
            previous = now
            if out is not None:
                out.write(frame)
            if show:
                cv2.imshow("SwingVision", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cap.release()
    if out is not None:
        out.release()
    if show:
        cv2.destroyAllWindows()
    return tracker.get_swing()


def fill_gaps(values: np.array) -> np.array:
    values = np.array(values, dtype=float)
    columns = values.reshape(len(values), -1)
    index = np.arange(len(values))
    for column in columns.T:
        valid = ~np.isnan(column)
        if valid.any():
            column[~valid] = np.interp(index[~valid], index[valid], column[valid])
    return columns.reshape(values.shape)


def smooth(values: np.array, window: int) -> np.array:
    # Moving average: the derivative amplifies the noise, so the trajectories are smoothed before deriving
    if window < 3:
        return values
    kernel = np.ones(window) / window
    columns = values.reshape(len(values), -1)
    padded = np.pad(columns, ((window // 2, window // 2), (0, 0)), mode="edge")
    smoothed = np.column_stack([np.convolve(column, kernel, mode="valid") for column in padded.T])
    return smoothed.reshape(values.shape)


def prepare_trajectories(swing: dict):
    window = int(round(config.SMOOTH_SECONDS * swing["fps"]))
    window += 1 - window % 2   # odd, so the average is centred
    t = swing["t"]
    swing["hands"] = smooth(fill_gaps(swing["grip"]), window)
    swing["head"] = smooth(fill_gaps(swing["cap"]), window)
    swing["club_head"] = smooth(fill_gaps(swing["club"]), window)
    swing["body"] = smooth(fill_gaps(swing["centre"]), window)
    swing["body_tilt"] = smooth(fill_gaps(swing["tilt"]), window)
    swing["hands_speed"] = np.linalg.norm(np.gradient(swing["hands"], t, axis=0), axis=1)
    swing["club_speed"] = np.linalg.norm(np.gradient(swing["club_head"], t, axis=0), axis=1)
    # Continuous angle of the club: unwrap removes the jumps between -180 and 180 degrees
    rotation = np.full(len(t), np.nan)
    valid = ~np.isnan(swing["angle"])
    rotation[valid] = np.unwrap(np.radians(swing["angle"][valid]))
    swing["rotation"] = smooth(fill_gaps(rotation), window)


def detect_positions(swing: dict):
    """Returns {"P1": index, ...} with the index in the arrays of the swing (None if a position is not found)"""
    if len(swing["t"]) < 10 or np.isnan(swing["grip"]).all() or np.isnan(swing["club"]).all() or np.isnan(swing["cap"]).all():
        return None
    prepare_trajectories(swing)
    t, hands, speed = swing["t"], swing["hands"], swing["hands_speed"]

    # The downswing is the fastest movement of the hands
    fastest = int(np.argmax(speed))
    still = speed < config.STILL_SPEED_RATIO * speed[fastest]

    # P4 (top): highest point of the hands before the fastest moment (y grows downwards)
    start = int(np.searchsorted(t, t[fastest] - config.MAX_BACKSWING_SECONDS))
    heights = hands[start:fastest + 1, 1]
    # The hands hardly move at the top: the middle of the frames within 1 px of the highest point is taken
    p4 = start + int(np.median(np.where(heights <= heights.min() + 1)[0]))

    # P1 (address): walking back from the top through the backswing, until the hands are still
    i = p4
    while i > 0 and still[i]:
        i -= 1
    while i > 0 and not still[i]:
        i -= 1
    p1 = i
    if p1 == 0 or p4 - p1 < 3:
        return None

    # P7 (impact): the hands come back to where they were at address
    end = int(np.searchsorted(t, t[p4] + config.MAX_DOWNSWING_SECONDS))
    distance = np.linalg.norm(hands[p4 + 1:end + 1] - hands[p1], axis=1)
    if len(distance) == 0:
        return None
    p7 = p4 + 1 + int(np.argmin(distance))

    # P2 and P6 (club horizontal): the sine of the angle changes sign when the club head passes the height of the hands
    sine = np.sin(swing["rotation"])
    up = [i for i in range(p1 + 1, p4 + 1) if sine[i - 1] < 0 <= sine[i]]
    down = [i for i in range(p4 + 1, p7 + 1) if sine[i - 1] >= 0 > sine[i]]
    closest = lambda i: i if abs(sine[i]) < abs(sine[i - 1]) else i - 1
    p2 = closest(up[0]) if up else None       # first time it goes above the hands in the backswing
    p6 = closest(down[-1]) if down else None  # last time it goes below the hands in the downswing
    return {"P1": p1, "P2": p2, "P4": p4, "P6": p6, "P7": p7}


def compute_measures(swing: dict, positions: dict) -> dict:
    # Positions are relative to the club head at address and divided by the height of the player (club head -
    # cap at address), so that players of different size or at a different distance can be compared
    t = swing["t"]
    p1, p4, p7 = positions["P1"], positions["P4"], positions["P7"]
    origin = swing["club_head"][p1]
    height = origin[1] - swing["head"][p1][1]
    swing["origin"], swing["height"] = origin, height

    measures = {}
    for name, i in positions.items():
        if i is None:
            continue
        measures[f"{name}.hands_x"] = (swing["hands"][i][0] - origin[0]) / height
        measures[f"{name}.hands_y"] = (origin[1] - swing["hands"][i][1]) / height
        measures[f"{name}.head_x"] = (swing["head"][i][0] - swing["head"][p1][0]) / height
        measures[f"{name}.head_y"] = (swing["head"][p1][1] - swing["head"][i][1]) / height
        measures[f"{name}.club"] = np.degrees(abs(swing["rotation"][i] - swing["rotation"][p1]))
        measures[f"{name}.sway"] = (swing["body"][i][0] - swing["body"][p1][0]) / height
        measures[f"{name}.tilt"] = swing["body_tilt"][i]

    measures["tempo"] = (t[p4] - t[p1]) / (t[p7] - t[p4])
    # Maximum speed in the downswing (a few frames after the impact are included)
    end = min(int(np.searchsorted(t, t[p7] + 0.05)), len(t) - 1)
    peak = p4 + int(np.argmax(swing["club_speed"][p4:end + 1]))
    measures["club_peak"] = (t[peak] - t[p7]) * 1000   # ms, negative = before the impact
    if swing["mm_per_px"] is not None:
        to_metres = swing["mm_per_px"] / 1000
        measures["hands_speed"] = swing["hands_speed"][p4:end + 1].max() * to_metres
        measures["club_speed"] = swing["club_speed"][peak] * to_metres
    return {key: float(value) for key, value in measures.items()}


def get_position_images(swing: dict, positions: dict) -> dict:
    images = {}
    for name, i in positions.items():
        if i is None:
            continue
        frame = cv2.imdecode(swing["jpg"][i], cv2.IMREAD_COLOR)
        foreground, silhouette = segment_player(frame, swing["background"])
        images[name] = (frame, foreground, silhouette)
    return images


def get_tracking_quality(swing: dict, positions: dict) -> dict:
    detected = swing["detected"][positions["P1"]:positions["P7"] + 1]
    return {name: 100 * detected[:, k].mean() for k, name in enumerate(config.MARKERS)}
