import cv2
import numpy as np
from typing import List

import config
from utils import draw_text


def get_color_masks(hsv: np.array) -> dict:
    masks = {}
    for color, ranges in config.FLAG_HUES.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in ranges:
            in_range = cv2.inRange(hsv, (low, config.COLOR_MIN_SATURATION, config.COLOR_MIN_VALUE), (high, 255, 255))
            mask = cv2.bitwise_or(mask, in_range)
        masks[color] = mask
    masks["black"] = cv2.inRange(hsv, (0, 0, 0), (179, config.BLACK_MAX_SATURATION, config.BLACK_MAX_VALUE))
    return masks


def get_triangularity(contour: np.array) -> float:
    # A pennant is a triangle: it fills most of the smallest triangle that encloses it (a circle fills 60 %)
    triangle_area, _ = cv2.minEnclosingTriangle(contour)
    return cv2.contourArea(contour) / max(triangle_area, 1)


def get_paper_mask(hsv: np.array) -> np.array:
    # White paper is bright and has no colour (skin, hair and clothes are darker or more saturated)
    return (hsv[:, :, 1] < config.PAPER_MAX_SATURATION) & (hsv[:, :, 2] > config.PAPER_MIN_VALUE)


def get_paper_around(paper: np.array, contour: np.array) -> float:
    # The pennant is drawn on paper: fraction of the pixels around it that are paper
    blob = np.zeros(paper.shape, np.uint8)
    cv2.drawContours(blob, [contour], -1, 255, -1)
    width = int(max(3, 0.25 * np.sqrt(cv2.contourArea(contour))))
    outer = cv2.dilate(blob, np.ones((2 * width + 1, 2 * width + 1), np.uint8))
    inner = cv2.dilate(blob, np.ones((5, 5), np.uint8))   # the edge of the pennant is mixed with the paper
    return float(paper[(outer > 0) & (inner == 0)].mean())


def find_pole(gray: np.array, paper: np.array, contour: np.array):
    """Looks for the pole: a dark line drawn on the paper, longer than the pennant, that touches it.
    Returns (x1, y1, x2, y2) or None"""
    x, y, w, h = cv2.boundingRect(contour)
    size = max(w, h)
    margin = int(2.5 * size)
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    roi = gray[y0:y + h + margin, x0:x + w + margin]
    # The pole is thin and drawn with a pen: it is darker than its neighbourhood, whatever the light
    ink = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, config.POLE_BLOCK_SIZE, config.POLE_OFFSET)
    lines = cv2.HoughLinesP(ink, 1, np.pi / 180, 25, minLineLength=config.MIN_POLE_LENGTH * size, maxLineGap=6)
    if lines is None:
        return None

    pole, pole_length = None, 0
    for xa, ya, xb, yb in lines.reshape(-1, 4) + np.array([x0, y0, x0, y0]):
        points = np.linspace((xa, ya), (xb, yb), 20)
        # The line must touch the bounding box of the pennant (the edges of the card are further away)
        distance = min(max(x - px, 0, px - x - w) + max(y - py, 0, py - y - h) for px, py in points)
        if distance >= 6:
            continue
        # A line drawn on the paper has paper on both sides (the outline of the hair or of a hand does not)
        length = np.hypot(xb - xa, yb - ya)
        normal = np.array([ya - yb, xb - xa]) / length
        sides = []
        for offset in (4, -4):
            beside = np.round(points + offset * normal).astype(int)
            columns = np.clip(beside[:, 0], 0, paper.shape[1] - 1)
            rows = np.clip(beside[:, 1], 0, paper.shape[0] - 1)
            sides.append(paper[rows, columns].mean())
        if min(sides) >= config.MIN_PAPER_BESIDE_POLE and length > pole_length:
            pole, pole_length = (int(xa), int(ya), int(xb), int(yb)), length
    return pole


def find_candidates(frame: np.array) -> List[dict]:
    """Every blob with the colour and the size of a pennant, with the values of the three checks of a flag"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    paper = get_paper_mask(hsv)
    min_area, max_area = config.PENNANT_AREA[0] * gray.size, config.PENNANT_AREA[1] * gray.size

    candidates = []
    for color, mask in get_color_masks(hsv).items():
        # The opening removes the pole and the lines of the drawing, only the filled pennant is left
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not min_area < cv2.contourArea(contour) < max_area:
                continue
            candidate = {"color": color, "contour": contour, "triangularity": get_triangularity(contour),
                         "paper": get_paper_around(paper, contour), "pole": None}
            if is_pennant(candidate):
                candidate["pole"] = find_pole(gray, paper, contour)
            candidates.append(candidate)
    return candidates


def is_pennant(candidate: dict) -> bool:
    return candidate["triangularity"] > config.MIN_TRIANGULARITY and candidate["paper"] > config.MIN_PAPER_AROUND


def detect_flag(frame: np.array):
    """Detects the flag shown to the camera: a coloured triangle on white paper with a pole next to it

    Returns:
        tuple: color (str or None), pennant (contour or None), pole (x1, y1, x2, y2 or None)
    """
    flags = [candidate for candidate in find_candidates(frame) if is_pennant(candidate) and candidate["pole"] is not None]
    if len(flags) == 0:
        return None, None, None
    flag = max(flags, key=lambda candidate: cv2.contourArea(candidate["contour"]))
    return flag["color"], flag["contour"], flag["pole"]


def draw_detection(frame: np.array, color, pennant, pole) -> np.array:
    if pennant is None:
        return frame
    cv2.drawContours(frame, [pennant], -1, (0, 255, 255), 2)
    cv2.line(frame, pole[:2], pole[2:], (255, 0, 255), 2)
    x, y, w, h = cv2.boundingRect(pennant)
    draw_text(frame, color, (x, max(y - 10, 20)), (0, 255, 0))
    return frame


class SequenceDecoder:
    """state - "waiting" (sequence not complete), "granted" (access for self.user) or "denied" (wrong sequence)"""

    def __init__(self, users: dict = config.USERS, accept_frames: int = config.ACCEPT_FRAMES,
                 release_frames: int = config.RELEASE_FRAMES, timeout: float = config.SEQUENCE_TIMEOUT):
        self.users = users
        self.accept_frames = accept_frames
        self.release_frames = release_frames
        self.timeout = timeout
        self.length = len(next(iter(users.values())))
        self.reset()

    def reset(self):
        self.sequence = []
        self.state = "waiting"
        self.user = None
        self.candidate = None    # flag seen in the last frames
        self.count = 0           # consecutive frames with the candidate
        self.empty_count = 0     # consecutive frames without any flag
        self.released = True     # True when the previous card has already been taken away
        self.last_time = None    # time of the last accepted flag

    def update(self, color, now: float) -> str:
        if self.state == "granted":
            return self.state

        # The sequence is forgotten if too much time passes without showing a flag
        if self.sequence and now - self.last_time > self.timeout:
            self.reset()

        if color is None:
            # One frame without flag can be a detection failure: the card has been taken away after several
            self.empty_count += 1
            if self.empty_count >= self.release_frames:
                self.released = True
                self.candidate = None
                self.count = 0
            return self.state

        self.empty_count = 0
        if color == self.candidate:
            self.count += 1
        else:
            self.candidate = color
            self.count = 1

        # A flag is accepted only once: it must be stable and the previous card must have been taken away
        if self.released and self.count >= self.accept_frames:
            self.state = "waiting"
            self.sequence.append(color)
            self.released = False
            self.last_time = now
            if len(self.sequence) == self.length:
                self.check_sequence()
        return self.state

    def check_sequence(self):
        for user, code in self.users.items():
            if self.sequence == code:
                self.state = "granted"
                self.user = user
                return
        self.sequence = []
        self.state = "denied"

    def draw(self, frame: np.array) -> np.array:
        if self.state == "granted":
            draw_text(frame, f"Acceso concedido: {self.user}", (10, 60), (0, 255, 0), 0.8)
        elif self.state == "denied":
            draw_text(frame, "Acceso denegado: secuencia incorrecta", (10, 60), (0, 0, 255), 0.8)
        else:
            draw_text(frame, f"Secuencia {len(self.sequence)}/{self.length}: {' '.join(self.sequence)}", (10, 60), (255, 255, 255), 0.8)
        return frame
