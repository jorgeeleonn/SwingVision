import os
import json
import cv2
import numpy as np
from typing import List
import matplotlib.pyplot as plt

import config
from utils import draw_text
from swing import analyse_video, detect_positions, compute_measures, get_position_images

# name: (label, kind of tolerance, unit, text if the value is higher than the reference, text if it is lower)
MEASURE_INFO = {
    "hands_x": ("manos (horizontal)", "position", "cm", "más a la derecha", "más a la izquierda"),
    "hands_y": ("manos (altura)", "position", "cm", "más altas", "más bajas"),
    "head_x": ("cabeza (horizontal)", "position", "cm", "más a la derecha", "más a la izquierda"),
    "head_y": ("cabeza (altura)", "position", "cm", "más alta", "más baja"),
    "club": ("giro del palo", "angle", "grados", "más giro", "menos giro"),
    "sway": ("desplazamiento lateral", "position", "cm", "más a la derecha", "más a la izquierda"),
    "tilt": ("inclinación del cuerpo", "angle", "grados", "más a la derecha", "más a la izquierda"),
    "tempo": ("tempo", "tempo", "", "subida más lenta", "subida más rápida"),
    "hands_speed": ("velocidad de las manos", "hands_speed", "m/s", "más rápidas", "más lentas"),
    "club_speed": ("velocidad del palo", "club_speed", "m/s", "más rápido", "más lento"),
    "club_peak": ("velocidad máxima", "time", "ms", "más tarde", "antes"),
}
POSITION_NAMES = {"P1": "address", "P2": "palo horizontal (subida)", "P4": "cima",
                  "P6": "palo horizontal (bajada)", "P7": "impacto"}
LEVEL_COLORS = {"bien": (0, 200, 0), "a mejorar": (0, 190, 255), "mal": (0, 0, 255)}   # BGR
LEVEL_POINTS = {"bien": 1.0, "a mejorar": 0.5, "mal": 0.0}


def build_reference(swings_measures: List) -> dict:
    reference = {}
    for key in swings_measures[0]:
        values = [measures[key] for measures in swings_measures if key in measures]
        if len(values) == len(swings_measures):
            reference[key] = (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    return reference


def register_reference(videopaths: List, calib: dict, folder: str = config.REFERENCE_PATH):
    os.makedirs(folder, exist_ok=True)
    swings_measures = []
    for k, videopath in enumerate(videopaths):
        swing = analyse_video(videopath, calib)
        positions = detect_positions(swing) if swing is not None else None
        if positions is None or None in positions.values():
            print(f"{os.path.basename(videopath)}: swing not detected, video ignored")
            continue
        swings_measures.append(compute_measures(swing, positions))
        print(f"{os.path.basename(videopath)}: P frames {positions}")
        if len(swings_measures) == 1:
            # The first valid swing is the one drawn as the ghost
            images = get_position_images(swing, positions)
            for name, (frame, foreground, silhouette) in images.items():
                cv2.imwrite(os.path.join(folder, f"{name}.png"), silhouette)
            cv2.imwrite(os.path.join(folder, "strobe.png"), make_strobe(images, swing["background"]))
            drawing = {"origin": swing["origin"].tolist(), "height": float(swing["height"]),
                       "points": {name: {"grip": swing["hands"][i].tolist(), "club": swing["club_head"][i].tolist()}
                                  for name, i in positions.items()},
                       "curve": get_speed_curve(swing, positions)}

    data = {"n": len(swings_measures), "measures": build_reference(swings_measures), **drawing}
    with open(os.path.join(folder, "reference.json"), "w") as f:
        json.dump(data, f, indent=2)
    print(f"Reference saved in {folder} ({data['n']} swings)")
    return swings_measures


def load_reference(folder: str = config.REFERENCE_PATH):
    path = os.path.join(folder, "reference.json")
    if not os.path.exists(path):
        print("Warning: there is no reference, run register_reference.py first")
        return None
    with open(path) as f:
        reference = json.load(f)
    reference["masks"] = {name: cv2.imread(os.path.join(folder, f"{name}.png"), cv2.IMREAD_GRAYSCALE) for name in config.POSITIONS}
    reference["strobe"] = cv2.imread(os.path.join(folder, "strobe.png"))
    return reference


def evaluate(value: float, mean: float, std: float, kind: str, k_verde: float = config.K_VERDE, k_rojo: float = config.K_ROJO):
    green = max(k_verde * std, config.TOL_MIN[kind])   # up to here: bien
    red = green * k_rojo / k_verde                     # from here on: mal
    difference = value - mean
    if abs(difference) <= green:
        level = "bien"
    elif abs(difference) <= red:
        level = "a mejorar"
    else:
        level = "mal"
    return level, difference, abs(difference) / green


def compare(measures: dict, reference: dict, cm_per_unit: float = None) -> List:
    """cm_per_unit - centimetres of one height of the player, to write the position differences in cm"""
    keys = [(position, name) for position in config.POSITIONS for name in config.MEASURES[position]]
    keys += [("", name) for name in config.GLOBAL_MEASURES]
    rows = []
    for position, name in keys:
        key = f"{position}.{name}" if position else name
        if key not in measures or key not in reference["measures"]:
            continue
        label, kind, unit, higher, lower = MEASURE_INFO[name]
        mean, std = reference["measures"][key]
        level, difference, z = evaluate(measures[key], mean, std, kind)

        # Text for the player, for example "8 cm más bajas"
        amount = abs(difference)
        if kind == "position":
            amount, unit = (amount * cm_per_unit, "cm") if cm_per_unit is not None else (amount * 100, "% de la altura")
        text = f"{amount:.1f} {unit} {higher if difference > 0 else lower}"
        if name == "tempo":
            text = f"{measures[key]:.1f} frente a {mean:.1f}, {higher if difference > 0 else lower}"
        rows.append({"position": position, "name": name, "label": label, "value": measures[key], "mean": mean,
                     "difference": difference, "z": z, "level": level, "text": text})
    return rows


def summarise(rows: List) -> dict:
    counts = {level: sum(row["level"] == level for row in rows) for level in LEVEL_POINTS}
    score = 10 * sum(LEVEL_POINTS[row["level"]] for row in rows) / max(len(rows), 1)
    faults = sorted([row for row in rows if row["level"] != "bien"], key=lambda row: row["z"], reverse=True)
    first = next((row for row in rows if row["level"] != "bien" and row["position"]), None)
    return {"score": score, "counts": counts, "priorities": faults[:3], "first": first}


def report_lines(rows: List, summary: dict) -> List:
    # List of (line, level), so that the report can be printed or drawn with colours
    counts = summary["counts"]
    lines = [(f"Nota: {summary['score']:.1f} / 10   ({counts['bien']} bien, {counts['a mejorar']} a mejorar, {counts['mal']} mal)", None)]
    position = None
    for row in rows:
        if row["position"] != position:
            position = row["position"]
            lines.append((f"{position} {POSITION_NAMES[position]}" if position else "Swing completo", None))
        detail = "" if row["level"] == "bien" else f" ({row['text']})"
        lines.append((f"   {row['label']}: {row['level']}{detail}", row["level"]))
    lines.append(("Prioridades", None))
    for k, row in enumerate(summary["priorities"]):
        where = f"{row['position']} " if row["position"] else ""
        lines.append((f"   {k + 1}. {where}{row['label']}: {row['text']} (z = {row['z']:.1f})", row["level"]))
    if summary["first"] is not None:
        row = summary["first"]
        lines.append((f"El fallo empieza en {row['position']} ({row['label']}: {row['text']})", row["level"]))
    return lines


def print_report(rows: List, summary: dict):
    for line, level in report_lines(rows, summary):
        print(line)


def draw_report(rows: List, summary: dict, width: int = 760) -> np.array:
    lines = report_lines(rows, summary)
    image = np.full((28 + 24 * len(lines), width, 3), 30, np.uint8)
    for k, (line, level) in enumerate(lines):
        color = LEVEL_COLORS[level] if level is not None else (255, 255, 255)
        draw_text(image, line, (12, 32 + 24 * k), color, 0.55, 1)
    return image


def draw_ghost(frame: np.array, position: str, swing: dict, reference: dict, rows: List) -> np.array:
    image = frame.copy()
    # The reference is scaled to the size of the player and moved so that both club heads at address coincide
    scale = swing["height"] / reference["height"]
    M = np.float32([[scale, 0, swing["origin"][0] - scale * reference["origin"][0]],
                    [0, scale, swing["origin"][1] - scale * reference["origin"][1]]])
    mask = cv2.warpAffine(reference["masks"][position], M, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ghost = image.copy()
    cv2.drawContours(ghost, contours, -1, (255, 255, 255), -1)
    cv2.addWeighted(ghost, 0.45, image, 0.55, 0, dst=image)
    cv2.drawContours(image, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    points = np.float32([reference["points"][position]["grip"], reference["points"][position]["club"]])
    grip, club = cv2.transform(points.reshape(-1, 1, 2), M).reshape(-1, 2)
    cv2.line(image, (int(grip[0]), int(grip[1])), (int(club[0]), int(club[1])), (255, 255, 255), 2, cv2.LINE_AA)

    draw_text(image, f"{position} {POSITION_NAMES[position]} (blanco = referencia)", (10, 30), (255, 255, 255), 0.7)
    lines = [row for row in rows if row["position"] == position]
    for k, row in enumerate(lines):
        detail = "" if row["level"] == "bien" else f" ({row['text']})"
        draw_text(image, f"{row['label']}: {row['level']}{detail}", (10, 58 + 22 * k), LEVEL_COLORS[row["level"]], 0.5, 1)
    return image


def make_strobe(images: dict, background: np.array) -> np.array:
    strobe = background.copy()
    for name in config.POSITIONS:
        if name not in images:
            continue
        frame, foreground, silhouette = images[name]
        # The foreground mask is used because it also contains the club
        mask = cv2.dilate(foreground, np.ones((3, 3), np.uint8))
        strobe[mask > 0] = frame[mask > 0]
    return strobe


def get_speed_curve(swing: dict, positions: dict) -> dict:
    # Time in ms, 0 = impact
    t = swing["t"]
    first, p7 = positions["P1"], positions["P7"]
    last = min(int(np.searchsorted(t, t[p7] + 0.1)), len(t) - 1)
    unit = swing["mm_per_px"] / 1000 if swing["mm_per_px"] is not None else 1 / swing["height"]
    return {"t": ((t[first:last + 1] - t[p7]) * 1000).tolist(),
            "club": (swing["club_speed"][first:last + 1] * unit).tolist(),
            "hands": (swing["hands_speed"][first:last + 1] * unit).tolist(),
            "positions": {name: float((t[i] - t[p7]) * 1000) for name, i in positions.items() if i is not None}}


def plot_speed(swing: dict, positions: dict, reference: dict, save_path: str = None, show: bool = True):
    curve = get_speed_curve(swing, positions)
    plt.figure(figsize=(10, 4))
    plt.plot(curve["t"], curve["club"], color="tab:blue", label="Palo (jugador)")
    plt.plot(curve["t"], curve["hands"], color="tab:orange", label="Manos (jugador)")
    if reference is not None:
        plt.plot(reference["curve"]["t"], reference["curve"]["club"], "--", color="tab:blue", alpha=0.6, label="Palo (referencia)")
        plt.plot(reference["curve"]["t"], reference["curve"]["hands"], "--", color="tab:orange", alpha=0.6, label="Manos (referencia)")
    for name, time_ms in curve["positions"].items():
        plt.axvline(time_ms, color="gray", linestyle=":")
        plt.text(time_ms, plt.ylim()[1] * 0.95, name, ha="center")
    plt.xlabel("Tiempo respecto al impacto [ms]")
    plt.ylabel("Velocidad [m/s]" if swing["mm_per_px"] is not None else "Velocidad [alturas/s]")
    plt.title("Velocidad del palo y de las manos")
    plt.legend(loc="center left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def save_results(folder: str, swing: dict, positions: dict, reference: dict, rows: List, summary: dict) -> dict:
    os.makedirs(folder, exist_ok=True)
    images = get_position_images(swing, positions)
    outputs = {"report": draw_report(rows, summary)}
    for name, (frame, foreground, silhouette) in images.items():
        outputs[name] = draw_ghost(frame, name, swing, reference, rows)
    # Player on the left, reference on the right (resized to the same height)
    strobe = make_strobe(images, swing["background"])
    reference_strobe = cv2.resize(reference["strobe"], (int(reference["strobe"].shape[1] * strobe.shape[0] / reference["strobe"].shape[0]), strobe.shape[0]))
    outputs["strobe"] = np.hstack([draw_text(strobe, "Jugador", (10, 30)), draw_text(reference_strobe, "Referencia", (10, 30))])
    for name, image in outputs.items():
        cv2.imwrite(os.path.join(folder, f"{name}.png"), image)
    plot_speed(swing, positions, reference, os.path.join(folder, "speed.png"), show=False)
    with open(os.path.join(folder, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(line for line, level in report_lines(rows, summary)))
    return outputs
