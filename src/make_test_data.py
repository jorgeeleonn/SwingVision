"""
Synthetic test data for SwingVision.

The real project uses our own recordings. This script only draws fake data (a chessboard, the flag cards
and a "golfer" made of simple shapes) with KNOWN values, so that every block can be tested before recording
anything. Everything is written in ../data and can be deleted once the real recordings are there.
"""
import os
import json
import cv2
import numpy as np

import config

rng = np.random.default_rng(0)

# Synthetic camera: 1280x720, focal 1000 px, a little radial distortion
WIDTH, HEIGHT, FPS = 1280, 720, 240
K = np.array([[1000.0, 0, 640], [0, 1000.0, 360], [0, 0, 1]])
DIST = np.array([-0.12, 0.04, 0.0, 0.0, 0.0])
MM_PER_PX = 3.9                      # scale in the plane of the player (distance = 3.9 m)
PPM = 1000 / MM_PER_PX               # pixels per metre
GROUND_Y = 680                       # image row of the floor under the player

FLAG_BGR = {"red": (40, 40, 215), "green": (60, 165, 40), "blue": (190, 90, 30), "black": (25, 25, 25)}
MARKER_BGR = {"cap": (200, 40, 230), "grip": (0, 230, 255), "club": (255, 220, 0)}


def add_noise(img, sigma=3):
    noise = np.zeros(img.shape, np.int16)
    cv2.randn(noise, 0, sigma)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def get_distortion_maps():
    """Maps that turn an ideal image into the image the (distorted) camera would record"""
    xs, ys = np.meshgrid(np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(points, K, DIST, P=K).reshape(HEIGHT, WIDTH, 2)
    return ideal[:, :, 0].copy(), ideal[:, :, 1].copy()


MAP_X, MAP_Y = get_distortion_maps()


def distort(img):
    return cv2.remap(img, MAP_X, MAP_Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- Chessboard images
def draw_board(px_per_mm=4, margin=25):
    cols, rows = config.PATTERN_SIZE[0] + 1, config.PATTERN_SIZE[1] + 1
    size = config.SQUARE_SIZE * px_per_mm
    board = np.full((rows * size + 2 * margin * px_per_mm, cols * size + 2 * margin * px_per_mm), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin * px_per_mm + r * size, margin * px_per_mm + c * size
                board[y:y + size, x:x + size] = 0
    # Texture pixels -> millimetres (origin at the first inner corner)
    to_mm = np.array([[1 / px_per_mm, 0, -margin - config.SQUARE_SIZE], [0, 1 / px_per_mm, -margin - config.SQUARE_SIZE], [0, 0, 1]])
    return board, to_mm


def project_board(board, to_mm, rvec, tvec, background):
    R = cv2.Rodrigues(np.float32(rvec))[0]
    H = K @ np.column_stack([R[:, 0], R[:, 1], np.float32(tvec)]) @ to_mm
    warped = cv2.warpPerspective(cv2.cvtColor(board, cv2.COLOR_GRAY2BGR), H, (WIDTH, HEIGHT), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full(board.shape, 255, np.uint8), H, (WIDTH, HEIGHT))
    img = background.copy()
    img[mask > 127] = warped[mask > 127]
    corners = cv2.perspectiveTransform(np.float32([[[0, 0]], [[board.shape[1], 0]], [[board.shape[1], board.shape[0]]], [[0, board.shape[0]]]]), H)
    inside = np.all((corners[:, 0, 0] > 20) & (corners[:, 0, 0] < WIDTH - 20) & (corners[:, 0, 1] > 20) & (corners[:, 0, 1] < HEIGHT - 20))
    return img, inside


def make_calibration_images(scene):
    folder = os.path.join(config.DATA_PATH, "calibration")
    os.makedirs(folder, exist_ok=True)
    board, to_mm = draw_board()
    plain = cv2.GaussianBlur(add_noise(np.full((HEIGHT, WIDTH, 3), (120, 125, 130), np.uint8), 8), (0, 0), 3)
    saved = 0
    while saved < 15:
        rvec = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)]
        tvec = [rng.uniform(-420, 180), rng.uniform(-280, 80), rng.uniform(550, 950)]
        img, inside = project_board(board, to_mm, rvec, tvec, plain)
        if not inside:
            continue
        cv2.imwrite(os.path.join(folder, f"calib_{saved:03d}.jpg"), add_noise(distort(cv2.GaussianBlur(img, (3, 3), 0))))
        saved += 1
    # Board in the plane of the player, facing the camera, to measure the scale
    centre = [-(config.PATTERN_SIZE[0] - 1) * config.SQUARE_SIZE / 2, 150, MM_PER_PX * 1000]
    img, _ = project_board(board, to_mm, [0, 0, 0], centre, scene)
    cv2.imwrite(os.path.join(folder, "scale.jpg"), add_noise(distort(img), 2))
    print(f"Calibration: {saved} chessboard images + scale.jpg (true scale {MM_PER_PX} mm/px, true focal {K[0, 0]:.0f} px)")


# ---------------------------------------------------------------- Flag cards
def draw_flag_card(color, size=600):
    """Same design as our hand-made cards: square paper, thin dark pole, coloured pennant and the hole"""
    card = np.full((size, size, 3), (245, 248, 250), np.uint8)
    pen = (45, 45, 45)
    top, bottom, x = int(0.22 * size), int(0.80 * size), int(0.36 * size)
    cv2.line(card, (x, top), (x, bottom), pen, 8, cv2.LINE_AA)
    cv2.circle(card, (x, top), 11, pen, -1, cv2.LINE_AA)
    cv2.ellipse(card, (x, bottom), (30, 14), 0, 0, 360, pen, 4, cv2.LINE_AA)
    pennant = np.int32([[x + 4, top + 8], [0.72 * size, top + 0.10 * size], [x + 4, top + 0.20 * size]])
    cv2.fillPoly(card, [pennant], FLAG_BGR[color], cv2.LINE_AA)
    return card


def make_cards():
    folder = os.path.join(config.DATA_PATH, "cards")
    os.makedirs(folder, exist_ok=True)
    cards = {color: draw_flag_card(color) for color in FLAG_BGR}
    for color, card in cards.items():
        cv2.imwrite(os.path.join(folder, f"flag_{color}.png"), card)

    # Fake webcam scene: dark room, a lamp (bright but not a rectangle) and a hand holding the card
    w, h = 640, 480
    room = cv2.GaussianBlur(add_noise(np.full((h, w, 3), (70, 62, 58), np.uint8), 12), (0, 0), 4)
    cv2.ellipse(room, (560, 70), (60, 45), 0, 0, 360, (235, 240, 245), -1, cv2.LINE_AA)

    def show_card(color, tilt):
        frame = room.copy()
        card = cards[color]
        ch, cw = card.shape[:2]
        scale = 0.36
        centre = np.float32([310 + rng.uniform(-15, 15), 250 + rng.uniform(-10, 10)])
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), tilt, scale)
        M[:, 2] += centre - np.float32([cw / 2, ch / 2])
        warped = cv2.warpAffine(card, M, (w, h), flags=cv2.INTER_AREA)
        mask = cv2.warpAffine(np.full((ch, cw), 255, np.uint8), M, (w, h))
        # Hand (skin) below the card and thumb over it
        cv2.ellipse(frame, (int(centre[0]), int(centre[1] + 190)), (70, 90), 0, 0, 360, (140, 165, 215), -1, cv2.LINE_AA)
        frame[mask > 127] = warped[mask > 127]
        cv2.ellipse(frame, (int(centre[0] - 20), int(centre[1] + 128)), (16, 26), 20, 0, 360, (140, 165, 215), -1, cv2.LINE_AA)
        # Webcam look: slightly warm light, blur and noise
        frame = np.clip(frame * np.float32([0.92, 0.97, 1.0]), 0, 255).astype(np.uint8)
        return add_noise(cv2.GaussianBlur(frame, (3, 3), 0), 4)

    for color in cards:
        cv2.imwrite(os.path.join(folder, f"scene_{color}.png"), show_card(color, 8))
    cv2.imwrite(os.path.join(folder, "scene_empty.png"), add_noise(room, 4))

    # Videos of someone showing four cards: the code of jorge and a wrong code
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    sequences = {"sequence_jorge": config.USERS["jorge"], "sequence_wrong": ["red", "red", "green", "blue"]}
    for name, sequence in sequences.items():
        out = cv2.VideoWriter(os.path.join(folder, f"{name}.avi"), fourcc, 30, (w, h))
        for _ in range(15):
            out.write(add_noise(room, 4))
        for color in sequence:
            tilt = rng.uniform(-15, 15)
            for _ in range(30):
                out.write(show_card(color, tilt + rng.uniform(-2, 2)))
            for _ in range(15):
                out.write(add_noise(room, 4))
        out.release()
    print(f"Cards: {len(cards)} printable flags, {len(cards)} scenes and {len(sequences)} videos")


# ---------------------------------------------------------------- Swing videos
def make_scene():
    """Empty scene: wall, floor and some objects (two of them with the colours of the markers)"""
    scene = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    scene[:] = (196, 204, 208)
    scene[640:] = (70, 130, 80)
    cv2.rectangle(scene, (90, 470), (260, 640), (60, 80, 110), -1)        # box
    cv2.rectangle(scene, (930, 150), (1100, 270), (120, 100, 90), -1)     # picture
    cv2.rectangle(scene, (1150, 420), (1215, 500), (200, 40, 230), -1)    # magenta poster (same colour as the cap marker)
    cv2.circle(scene, (1020, 665), 22, (0, 230, 255), -1)                 # yellow bucket (same colour as the grip marker)
    texture = np.zeros((HEIGHT, WIDTH, 3), np.int16)
    cv2.randn(texture, 0, 10)
    texture = cv2.GaussianBlur(texture, (0, 0), 2)
    return np.clip(scene.astype(np.int16) + texture, 0, 255).astype(np.uint8)


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


def swing_times(p):
    u_imp = 0.5 ** (1 / p["gamma"])       # the down + follow-through phase reaches the impact at this fraction
    t = {}
    t["enter"] = p["t_bg"]
    t["still"] = t["enter"] + p["t_enter"]
    t["back"] = t["still"] + p["t_still"]
    t["top"] = t["back"] + p["t_back"]
    t["down"] = t["top"] + p["t_pause"]
    t["impact"] = t["down"] + p["t_down"]
    t["finish"] = t["down"] + p["t_down"] / u_imp
    t["end"] = t["finish"] + p["t_hold"]
    return t


def swing_state(t, p):
    """Position of every part of the golfer at time t (metres, x to the right, y up). None before he appears"""
    times = swing_times(p)
    if t < times["enter"]:
        return None
    k = p["size"]
    phi_top, beta_top = np.radians(p["phi_top"]), np.radians(p["beta_top"])
    beta_fin = beta_top + 2 * (np.radians(-90) - beta_top)

    walk = -3.3 * (1 - smoothstep((t - times["enter"]) / p["t_enter"]))
    sway = 0.0
    if t < times["back"]:
        phi, beta = 0.0, np.radians(-90)
    elif t < times["down"]:
        u = np.clip((t - times["back"]) / p["t_back"], 0, 1)
        s = u ** 2 * (3 - 2 * u)
        phi = phi_top * s
        beta = np.radians(-90) + (beta_top + np.radians(90)) * s
        sway = p["sway"] * s
    else:
        u = np.clip((t - times["down"]) / (times["finish"] - times["down"]), 0, 1)
        g = (1 - np.cos(np.pi * u ** p["gamma"])) / 2
        phi = phi_top * (1 - 2 * g)
        beta = beta_top + (beta_fin - beta_top) * (g - p["lag"] * np.sin(2 * np.pi * g))
        sway = p["sway"] * (1 - 2 * g) if g < 0.5 else 0.03 * (2 * g - 1)

    x0 = p["x0"] + walk
    state = {"k": k}
    state["feet"] = [(x0 - 0.22 * k, 0.0), (x0 + 0.22 * k, 0.0)]
    state["hip"] = (x0 + sway, 0.95 * k)
    state["shoulder"] = (x0 + 1.2 * sway, 1.42 * k)
    state["head"] = (x0 + 0.8 * sway, 1.62 * k)
    arm, club = 0.62 * k, 0.80 * k
    hands = (state["shoulder"][0] - arm * np.sin(phi), state["shoulder"][1] - arm * np.cos(phi))
    state["hands"] = hands
    state["beta"] = beta
    state["grip"] = (hands[0] + 0.12 * k * np.cos(beta), hands[1] + 0.12 * k * np.sin(beta))
    state["club"] = (hands[0] + club * np.cos(beta), hands[1] + club * np.sin(beta))
    return state


def to_px(point):
    return (WIDTH / 2 + point[0] * PPM, GROUND_Y - point[1] * PPM)


def draw_golfer(img, state):
    k = state["k"]
    px = lambda point: tuple(int(round(v)) for v in to_px(point))
    thick = lambda metres: max(1, int(round(metres * k * PPM)))
    trousers, shirt, skin, shaft = (60, 55, 50), (50, 45, 120), (140, 165, 215), (45, 45, 45)

    for foot in state["feet"]:
        cv2.line(img, px(state["hip"]), px(foot), trousers, thick(0.15), cv2.LINE_AA)
        cv2.ellipse(img, px((foot[0], 0.03 * k)), (thick(0.11), thick(0.045)), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
    hip, shoulder = state["hip"], state["shoulder"]
    torso = np.int32([px((hip[0] - 0.17 * k, hip[1])), px((hip[0] + 0.17 * k, hip[1])),
                      px((shoulder[0] + 0.21 * k, shoulder[1])), px((shoulder[0] - 0.21 * k, shoulder[1]))])
    cv2.fillPoly(img, [torso], shirt, cv2.LINE_AA)
    cv2.line(img, px(shoulder), px(state["head"]), skin, thick(0.09), cv2.LINE_AA)
    cv2.circle(img, px(state["head"]), thick(0.11), skin, -1, cv2.LINE_AA)
    for side in (-0.19, 0.19):
        cv2.line(img, px((shoulder[0] + side * k, shoulder[1] - 0.02 * k)), px(state["hands"]), shirt, thick(0.085), cv2.LINE_AA)
    cv2.line(img, px(state["hands"]), px(state["club"]), shaft, max(2, thick(0.012)), cv2.LINE_AA)

    # Markers: band on the cap, tape below the hands and tape on the club head
    angle = -np.degrees(state["beta"])
    boxes = {"cap": ((state["head"][0], state["head"][1] + 0.10 * k), (0.19, 0.07), 0),
             "grip": (state["grip"], (0.08, 0.055), angle),
             "club": (state["club"], (0.11, 0.075), angle)}
    for name, (centre, size, rotation) in boxes.items():
        box = cv2.boxPoints((to_px(centre), (size[0] * k * PPM, size[1] * k * PPM), rotation))
        cv2.fillPoly(img, [np.int32(np.round(box))], MARKER_BGR[name], cv2.LINE_AA)
    cv2.circle(img, px(state["hands"]), thick(0.05), skin, -1, cv2.LINE_AA)
    return img


def get_truth(p):
    """Frames of the P positions and other values of the model, to compare with what the system measures"""
    times = swing_times(p)
    t = np.arange(0, times["end"], 1 / (FPS * 8))
    states = [swing_state(ti, p) for ti in t]
    beta = np.array([np.degrees(s["beta"]) if s else np.nan for s in states])
    grip = np.array([s["grip"] if s else (np.nan, np.nan) for s in states])
    club = np.array([s["club"] if s else (np.nan, np.nan) for s in states])
    back = (t >= times["back"]) & (t < times["top"])
    down = (t >= times["down"]) & (t <= times["impact"])
    frame = lambda time: int(round(time * FPS))
    club_speed = np.linalg.norm(np.gradient(club, t, axis=0), axis=1)
    swing = (t >= times["down"]) & (t <= times["finish"])
    t_peak = t[swing][np.argmax(club_speed[swing])]
    height = (states[frame(times["back"]) * 8]["head"][1] + 0.10 * p["size"])  # cap marker over the club head at address
    return {
        "P1": frame(times["back"]),
        "P2": frame(t[back][np.argmin(np.abs(beta[back] + 180))]),
        "P4": frame(t[(t >= times["back"]) & (t <= times["impact"])][np.argmin(-grip[(t >= times["back"]) & (t <= times["impact"]), 1])]),
        "P6": frame(t[down][np.argmin(np.abs(beta[down] + 180))]),
        "P7": frame(times["impact"]),
        "tempo": p["t_back"] / p["t_down"],
        "club_speed": float(np.nanmax(club_speed[swing])),
        "hands_speed": float(np.nanmax(np.linalg.norm(np.gradient(grip, t, axis=0), axis=1)[swing])),
        "club_peak_ms": float((t_peak - times["impact"]) * 1000),
        "hands_top_height": float(np.nanmax(grip[:, 1]) / height),
        "sway_cm": p["sway"] * 100,
    }


def make_swing_video(path, p, scene):
    times = swing_times(p)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    n_frames = int(times["end"] * FPS)
    exposure = 0.25 / FPS   # the shutter is open a quarter of the time between frames
    for i in range(n_frames):
        t = i / FPS
        state = swing_state(t, p)
        if state is None:
            frame = scene.copy()
        else:
            # Motion blur: mean of the golfer drawn at three instants of the exposure
            later = swing_state(t + exposure, p)
            moved = np.hypot(*(np.subtract(to_px(later["club"]), to_px(state["club"]))))
            samples = 3 if moved > 1.5 else 1
            frame = np.zeros(scene.shape, np.float32)
            for j in range(samples):
                frame += draw_golfer(scene.copy(), swing_state(t + exposure * j / samples, p))
            frame = (frame / samples).astype(np.uint8)
        out.write(add_noise(distort(frame), 3))
    out.release()
    truth = get_truth(p)
    with open(os.path.splitext(path)[0] + "_truth.json", "w") as f:
        json.dump(truth, f, indent=2)
    return truth


def make_swings(scene):
    base = {"t_bg": 0.6, "t_enter": 1.0, "t_still": 0.45, "t_pause": 0.03, "t_hold": 0.3, "x0": 0.0, "size": 1.0,
            "t_back": 0.75, "t_down": 0.25, "phi_top": 150, "beta_top": -340, "gamma": 1.0, "lag": 0.06, "sway": -0.01}
    folder = os.path.join(config.DATA_PATH, "videos", "reference")
    os.makedirs(folder, exist_ok=True)
    for i in range(5):
        p = dict(base)
        p["t_back"] += rng.normal(0, 0.015)
        p["t_down"] += rng.normal(0, 0.005)
        p["phi_top"] += rng.normal(0, 1.5)
        p["beta_top"] += rng.normal(0, 3)
        p["sway"] += rng.normal(0, 0.004)
        p["x0"] += rng.normal(0, 0.01)
        truth = make_swing_video(os.path.join(folder, f"reference_{i + 1:02d}.mp4"), p, scene)
        print(f"reference_{i + 1:02d}.mp4:", {key: truth[key] for key in config.POSITIONS}, f"tempo {truth['tempo']:.2f}")

    # Player with known faults: lower hands at the top, more sway, quick backswing, early release.
    # He is also smaller and stands in another place, to check the normalisation
    folder = os.path.join(config.DATA_PATH, "videos", "player")
    os.makedirs(folder, exist_ok=True)
    p = dict(base, size=0.93, x0=0.18, t_back=0.62, t_down=0.27, phi_top=132, beta_top=-325, gamma=0.8, lag=0.0, sway=-0.06)
    truth = make_swing_video(os.path.join(folder, "swing_01.mp4"), p, scene)
    print("swing_01.mp4:", truth)


if __name__ == "__main__":
    scene = make_scene()
    make_cards()
    make_calibration_images(scene)
    make_swings(scene)
