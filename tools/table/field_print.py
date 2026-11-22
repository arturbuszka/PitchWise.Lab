"""Print file for the playfield vinyl: 1200 x 700 mm + bleed, with ArUco markers in the corners.

Run: python tools/table/field_print.py docs/foosball/drawings
The markers let calibration (plan step 3.4) find the playfield corners without manual clicking.
Marker centres in playfield mm are written next to the PDF, so the code can map them exactly.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

L, W = 1200, 700
BLEED = 3
LINE = 8
GREEN, WHITE = "#1f6b3a", "white"   # no yellow/orange anywhere: those are ball colours
GOAL_W = 200
PEN_D, PEN_W = 170, 420            # penalty area
BOX_D, BOX_W = 60, 260             # goal area, wider than the 200 mm goal
CIRCLE_R = 95
MARKER, QUIET, INSET = 40, 6, 12   # marker side, white border, distance from field edge
MARKER_IDS = [0, 1, 2, 3]          # DICT_4X4_50, order: bottom-left, bottom-right, top-right, top-left


def rect_line(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=WHITE, lw=0, joinstyle="miter"))
    # draw as filled strips so line width is exact in mm regardless of renderer
    for rx, ry, rw, rh in [(x, y, w, LINE), (x, y + h - LINE, w, LINE), (x, y, LINE, h), (x + w - LINE, y, LINE, h)]:
        ax.add_patch(Rectangle((rx, ry), rw, rh, color=WHITE, lw=0))


def marker(ax, cx, cy, marker_id, dictionary):
    bits = cv2.aruco.generateImageMarker(dictionary, marker_id, 6)  # 4x4 payload + 1-cell black border
    cell = MARKER / 6
    x0, y0 = cx - MARKER / 2, cy - MARKER / 2
    ax.add_patch(Rectangle((x0 - QUIET, y0 - QUIET), MARKER + 2 * QUIET, MARKER + 2 * QUIET, color=WHITE, lw=0))
    for r in range(6):
        for c in range(6):
            if bits[r, c] == 0:
                ax.add_patch(Rectangle((x0 + c * cell, y0 + (5 - r) * cell), cell, cell, color="black", lw=0))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=((L + 2 * BLEED) / 25.4, (W + 2 * BLEED) / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-BLEED, L + BLEED)
    ax.set_ylim(-BLEED, W + BLEED)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-BLEED, -BLEED), L + 2 * BLEED, W + 2 * BLEED, color=GREEN, lw=0))

    rect_line(ax, 0, 0, L, W)
    ax.add_patch(Rectangle((L / 2 - LINE / 2, 0), LINE, W, color=WHITE, lw=0))
    ax.add_patch(Circle((L / 2, W / 2), CIRCLE_R, fill=False, ec=WHITE, lw=LINE * 72 / 25.4))
    ax.add_patch(Circle((L / 2, W / 2), 8, color=WHITE, lw=0))
    for x, sign in ((0, 1), (L, -1)):
        px = x if sign > 0 else x - PEN_D
        bx = x if sign > 0 else x - BOX_D
        rect_line(ax, px, (W - PEN_W) / 2, PEN_D, PEN_W)
        rect_line(ax, bx, (W - BOX_W) / 2, BOX_D, BOX_W)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    c = INSET + QUIET + MARKER / 2
    centres = [(c, c), (L - c, c), (L - c, W - c), (c, W - c)]
    for marker_id, (mx, my) in zip(MARKER_IDS, centres):
        marker(ax, mx, my, marker_id, dictionary)

    fig.savefig(out / "boisko-folia.pdf")
    fig.savefig(out / "boisko-folia.png", dpi=40)
    plt.close(fig)
    (out / "boisko-markery.json").write_text(json.dumps({
        "dictionary": "DICT_4X4_50", "markerSizeMm": MARKER,
        "origin": "bottom-left corner of the playfield, x along length, y across (print orientation)",
        "markers": [{"id": i, "centreMm": [x, y]} for i, (x, y) in zip(MARKER_IDS, centres)],
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/foosball/drawings")
