import cv2

import config
from utils import draw_text, get_hsv_color_ranges
from calibration import prepare_frame
from main import open_source

# Test of the colours of the three markers with the real tape and the real camera.
# The pixels inside the range of a marker are painted with its colour: only the marker should be painted.
# 'c' freezes the frame and opens the tool of Lab 1 to find a range (it is printed in the terminal), 'q' quits


def main(source=config.CAMERA_INDEX):
    cap = open_source(source)
    if cap is None:
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = prepare_frame(frame, None)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        shown = frame.copy()
        for k, (name, (lower, upper)) in enumerate(config.MARKERS.items()):
            mask = cv2.inRange(hsv, lower, upper)
            shown[mask > 0] = config.MARKER_COLORS[name]
            draw_text(shown, f"{name}: {cv2.countNonZero(mask)} px, HSV {lower} - {upper}", (10, 30 + 25 * k), config.MARKER_COLORS[name], 0.6)
        cv2.imshow("Marker colours - 'q' to quit", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('c'):
            get_hsv_color_ranges(frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(config.CAMERA_INDEX)
