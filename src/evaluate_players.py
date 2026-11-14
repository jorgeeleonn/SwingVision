import os
import glob
import numpy as np

import config
from calibration import load_calibration
from swing import analyse_video, detect_positions, compute_measures
from comparison import load_reference, compare, summarise, save_results

# Analyses every recorded swing of every player and writes the table of results.
# The videos go in data/videos/player/<name of the player>/*.mp4


def evaluate_video(videopath: str, calib: dict, reference: dict, folder: str):
    swing = analyse_video(videopath, calib)
    positions = detect_positions(swing) if swing is not None else None
    if positions is None or None in positions.values():
        return None
    measures = compute_measures(swing, positions)
    cm_per_unit = swing["height"] * swing["mm_per_px"] / 10 if swing["mm_per_px"] is not None else None
    rows = compare(measures, reference, cm_per_unit)
    summary = summarise(rows)
    save_results(folder, swing, positions, reference, rows, summary)
    return summary


if __name__ == "__main__":
    calib = load_calibration()
    reference = load_reference()
    players = sorted(glob.glob(os.path.join(config.DATA_PATH, "videos", "player", "*", "")))
    lines = ["player,video,score,bien,a mejorar,mal,worst measure"]
    print(f"{'player':12s} {'video':16s} {'score':>5s} {'bien':>5s} {'a mejorar':>10s} {'mal':>4s}  worst measure")
    for player_folder in players if reference is not None else []:
        player = os.path.basename(os.path.dirname(player_folder))
        scores = []
        for videopath in sorted(glob.glob(os.path.join(player_folder, "*.mp4"))):
            video = os.path.splitext(os.path.basename(videopath))[0]
            summary = evaluate_video(videopath, calib, reference, os.path.join(config.RESULTS_PATH, player, video))
            if summary is None:
                print(f"{player:12s} {video:16s} swing not detected")
                continue
            counts = summary["counts"]
            worst = summary["priorities"][0] if summary["priorities"] else None
            worst_text = f"{worst['position']} {worst['label']}: {worst['text']}".strip() if worst is not None else "-"
            print(f"{player:12s} {video:16s} {summary['score']:5.1f} {counts['bien']:5d} {counts['a mejorar']:10d} {counts['mal']:4d}  {worst_text}")
            lines.append(f"{player},{video},{summary['score']:.2f},{counts['bien']},{counts['a mejorar']},{counts['mal']},{worst_text}")
            scores.append(summary["score"])
        if scores:
            print(f"{player:12s} {'MEAN':16s} {np.mean(scores):5.1f}   (min {min(scores):.1f}, max {max(scores):.1f}, {len(scores)} swings)\n")
            lines.append(f"{player},MEAN,{np.mean(scores):.2f},,,,")
    with open(os.path.join(config.RESULTS_PATH, "players.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Table saved in", os.path.join(config.RESULTS_PATH, "players.csv"))
