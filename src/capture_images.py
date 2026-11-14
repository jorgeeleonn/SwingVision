import os
import cv2

import config
from utils import draw_text
from main import open_source

# Saves the images of the chessboard needed by partA_calibration.ipynb, with the same camera and mode as the system.
# 's' saves a calibration image, 'e' saves scale.jpg (board where the player stands, facing the camera), 'q' quits


def main(source=config.CAMERA_INDEX):
    cap = open_source(source)
    if cap is None:
        return
    folder = os.path.join(config.DATA_PATH, "calibration")
    os.makedirs(folder, exist_ok=True)
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # FAST_CHECK only says if there is a board, so the video does not slow down when there is none
        found, corners = cv2.findChessboardCorners(frame, config.PATTERN_SIZE, None, cv2.CALIB_CB_FAST_CHECK)
        shown = frame.copy()
        cv2.drawChessboardCorners(shown, config.PATTERN_SIZE, corners, found)
        text = "Tablero detectado: 's' guarda" if found else "Tablero no detectado"
        draw_text(shown, f"{text} ({saved} guardadas, 'e' = escala)", (10, 30), (0, 255, 0) if found else (0, 0, 255), 0.8)
        cv2.imshow("Calibration images - 'q' to quit", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('s') and found:
            cv2.imwrite(os.path.join(folder, f"calib_{saved:03d}.jpg"), frame)
            saved += 1
        if key == ord('e') and found:
            cv2.imwrite(os.path.join(folder, "scale.jpg"), frame)
            print("scale.jpg saved")
    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} calibration images saved in {folder}")


if __name__ == "__main__":
    main(config.CAMERA_INDEX)
