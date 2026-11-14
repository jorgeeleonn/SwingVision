import time
import cv2

import config
from utils import draw_text
from calibration import prepare_frame
from security import find_candidates, is_pennant, detect_flag, draw_detection, SequenceDecoder
from main import open_source

# Test of the security block alone, with the real cards and the real camera.
# It shows the values the detector measures, to adjust the thresholds of config.py.
# 'r' deletes the sequence, 's' saves the current frame, 'q' quits


def describe(candidate: dict) -> str:
    # Values compared with MIN_TRIANGULARITY, MIN_PAPER_AROUND and MIN_POLE_LENGTH
    x, y, w, h = cv2.boundingRect(candidate["contour"])
    return (f"{candidate['color']} ({x},{y}) area {cv2.contourArea(candidate['contour']):.0f} | "
            f"triangularity {candidate['triangularity']:.2f} | paper {candidate['paper']:.2f} | "
            f"pole {'yes' if candidate['pole'] is not None else 'no'}")


def main(source=config.CAMERA_INDEX):
    cap = open_source(source)
    if cap is None:
        return
    decoder = SequenceDecoder()
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = prepare_frame(frame, None)   # no calibration here: only the detector is being tested
        original = frame.copy()
        color, pennant, pole = detect_flag(frame)
        decoder.update(color, time.time())

        # The candidates closest to being a pennant, to see which check fails
        candidates = sorted(find_candidates(frame), key=lambda c: c["triangularity"] * c["paper"], reverse=True)[:3]
        lines = [describe(candidate) for candidate in candidates] if candidates else ["No blob with the colour and size of a pennant"]
        for candidate in candidates:
            cv2.drawContours(frame, [candidate["contour"]], -1, (0, 255, 0) if is_pennant(candidate) else (0, 0, 255), 1)

        draw_detection(frame, color, pennant, pole)
        decoder.draw(frame)
        for k, line in enumerate(lines):
            draw_text(frame, line, (10, frame.shape[0] - 15 - 22 * k), scale=0.5, thickness=1)
        cv2.imshow("Security test - 'q' to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('r'):
            decoder.reset()
        if key == ord('s'):
            cv2.imwrite(f"security_{saved:02d}.png", original)
            print(f"security_{saved:02d}.png saved | detected: {color} |", " || ".join(lines))
            saved += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Same camera index that works in test.py
    main(config.CAMERA_INDEX)
