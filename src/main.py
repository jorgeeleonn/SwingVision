import os
import time
import cv2
import numpy as np

import config
from utils import draw_text
from calibration import load_calibration, prepare_frame, get_scale
from security import detect_flag, draw_detection, SequenceDecoder
from swing import SwingTracker, build_background, detect_positions, compute_measures, get_tracking_quality
from comparison import load_reference, compare, summarise, print_report, save_results

WINDOW = "SwingVision - 'q' para salir"
RECORD_SIZE = (960, 540)   # size of the video saved with --record


def open_source(source):
    if isinstance(source, int):
        # Use CAP_DSHOW for Windows to avoid warning messages
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_SIZE[1])
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open the source ({source}).")
        return None
    return cap


class FPSCounter:

    def __init__(self):
        self.previous = time.time()
        self.fps = 0

    def draw(self, frame: np.array) -> np.array:
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.previous, 1e-6)
        self.previous = now
        return draw_text(frame, f"FPS: {self.fps:.0f}", (10, 30), (0, 255, 0), 0.8)


def run_security(source, calib: dict, fps_counter: FPSCounter, out=None):
    cap = open_source(source)
    if cap is None:
        return None
    live = isinstance(source, int)
    decoder = SequenceDecoder()
    user = None
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = prepare_frame(frame, calib)
        color, pennant, pole = detect_flag(frame)
        # Live: real time. Video file: time of the frame (30 fps)
        decoder.update(color, time.time() if live else i / 30)
        i += 1

        draw_detection(frame, color, pennant, pole)
        decoder.draw(frame)
        draw_text(frame, "Muestra las 4 tarjetas de tu codigo ('r' = borrar)", (10, frame.shape[0] - 15), scale=0.5, thickness=1)
        fps_counter.draw(frame)
        cv2.imshow(WINDOW, frame)
        if out is not None:
            out.write(cv2.resize(frame, RECORD_SIZE))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            decoder.reset()
        if decoder.state == "granted":
            user = decoder.user
            cv2.waitKey(1000 if live else 1)
            break
    cap.release()
    return user


def analyse(tracker: SwingTracker, reference: dict, folder: str, show: bool = True):
    swing = tracker.get_swing()
    positions = detect_positions(swing)
    if positions is None or None in positions.values():
        print("No se ha detectado un swing completo:", positions)
        return False
    measures = compute_measures(swing, positions)
    quality = get_tracking_quality(swing, positions)
    print("Frames:", positions, "| markers seen [%]:", {name: round(value) for name, value in quality.items()})

    cm_per_unit = swing["height"] * swing["mm_per_px"] / 10 if swing["mm_per_px"] is not None else None
    rows = compare(measures, reference, cm_per_unit)
    summary = summarise(rows)
    print_report(rows, summary)
    outputs = save_results(folder, swing, positions, reference, rows, summary)
    print(f"Results saved in {folder}")
    if show:
        # One image after another: any key for the next one
        for name in ["report"] + config.POSITIONS + ["strobe"]:
            cv2.imshow(WINDOW, outputs[name])
            if cv2.waitKey(0) & 0xFF in (ord('q'), 27):
                break
        speed = cv2.imread(os.path.join(folder, "speed.png"))
        cv2.imshow(WINDOW, speed)
        cv2.waitKey(0)
    return True


def run_swing(source, calib: dict, reference: dict, user: str, fps_counter: FPSCounter, out=None, show_results: bool = True):
    cap = open_source(source)
    if cap is None:
        return
    live = isinstance(source, int)
    fps = config.VIDEO_FPS if config.VIDEO_FPS is not None else cap.get(cv2.CAP_PROP_FPS)
    n_background = int(config.BACKGROUND_SECONDS * (30 if live else fps))
    user_folder = os.path.join(config.RESULTS_PATH, user)
    os.makedirs(user_folder, exist_ok=True)
    n_swings = len([name for name in os.listdir(user_folder) if name.startswith("swing_")])

    tracker = None
    background_frames = [] if not live else None   # frames being collected for the background (None = not collecting)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = prepare_frame(frame, calib)
        t = time.time() if live else i / fps
        i += 1

        if background_frames is not None:
            background_frames.append(frame.copy())
            draw_text(frame, "Capturando el fondo: escena vacia", (10, 60), (0, 255, 255), 0.8)
            if len(background_frames) >= n_background:
                tracker = SwingTracker(build_background(background_frames), get_scale(calib, frame.shape[1]))
                background_frames = None
        elif tracker is None:
            draw_text(frame, "Deja la escena vacia y pulsa 'b'", (10, 60), (0, 255, 255), 0.8)
        else:
            tracker.update(frame, t)
            tracker.draw(frame)
            draw_text(frame, f"{user}: 'a' = analizar el swing, 'b' = nuevo fondo", (10, frame.shape[0] - 15), scale=0.5, thickness=1)

        fps_counter.draw(frame)
        cv2.imshow(WINDOW, frame)
        if out is not None:
            out.write(cv2.resize(frame, RECORD_SIZE))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('b'):
            tracker, background_frames = None, []
        if key == ord('a') and tracker is not None:
            if analyse(tracker, reference, os.path.join(user_folder, f"swing_{n_swings + 1:03d}")):
                n_swings += 1
            tracker.clear()
    cap.release()

    # A recorded swing is analysed when the video ends
    if not live and tracker is not None:
        analyse(tracker, reference, os.path.join(user_folder, f"swing_{n_swings + 1:03d}"), show_results)


def main(swing_source=config.CAMERA_INDEX, security_source=config.CAMERA_INDEX, record: str = config.RECORD_FILE, show_results: bool = True):
    calib = load_calibration()
    reference = load_reference()
    if reference is None:
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable window
    fps_counter = FPSCounter()

    out = None
    if record is not None:
        out = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*'XVID'), 30, RECORD_SIZE)

    try:
        user = run_security(security_source, calib, fps_counter, out)
        if user is not None:
            print(f"Access granted: {user}")
            run_swing(swing_source, calib, reference, user, fps_counter, out, show_results)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Live camera by default. To analyse a recorded swing, write its path in config.SWING_VIDEO
    main(swing_source=config.SWING_VIDEO if config.SWING_VIDEO is not None else config.CAMERA_INDEX,
         security_source=config.SECURITY_VIDEO if config.SECURITY_VIDEO is not None else config.CAMERA_INDEX)
