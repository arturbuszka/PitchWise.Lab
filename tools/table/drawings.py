"""Order drawings for the MDF parts of the table (docs/foosball/table-build.md).

Run: python tools/table/drawings.py docs/foosball/drawings
Writes one PDF (one part per page) and a PNG per part. All values in mm.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle

T = 18                      # MDF thickness
FIELD_L, FIELD_W = 1200, 700
BAND_H = 300
FIELD_TOP = 168             # playfield surface above band bottom edge
ROD_Z = 80                  # rod axis above playfield
ROD_PITCH, ROD_FIRST = 150, 75
HOLE_D = 24                 # depends on chosen bushing, see table-build.md
GOAL_W, GOAL_H = 200, 80

LONG_L = FIELD_L + 2 * T
ROD_X = [T + ROD_FIRST + i * ROD_PITCH for i in range(8)]
ROD_Y = FIELD_TOP + ROD_Z


def dim(ax, x1, y1, x2, y2, text, off, vertical=False):
    """Dimension line with arrows, offset from the measured edge."""
    if vertical:
        ax.annotate("", (x1 + off, y1), (x1 + off, y2), arrowprops=dict(arrowstyle="<->", lw=0.7))
        ax.text(x1 + off - 8, (y1 + y2) / 2, text, rotation=90, ha="right", va="center", fontsize=8)
        ax.plot([x1, x1 + off], [y1, y1], lw=0.3, color="grey")
        ax.plot([x2, x1 + off], [y2, y2], lw=0.3, color="grey")
    else:
        ax.annotate("", (x1, y1 + off), (x2, y1 + off), arrowprops=dict(arrowstyle="<->", lw=0.7))
        ax.text((x1 + x2) / 2, y1 + off + 6, text, ha="center", va="bottom", fontsize=8)
        ax.plot([x1, x1], [y1, y1 + off], lw=0.3, color="grey")
        ax.plot([x2, x2], [y2, y1 + off], lw=0.3, color="grey")


def page(title, w, h, qty, notes):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, lw=1.5))
    ax.set_aspect("equal")
    ax.axis("off")
    pad = max(w, h) * 0.12
    ax.set_xlim(-pad * 1.4, w + pad * 0.6)
    ax.set_ylim(-pad * 1.2, h + pad * 1.2)
    dim(ax, 0, h, w, h, f"{w}", pad * 0.45)
    dim(ax, 0, 0, 0, h, f"{h}", -pad * 0.9, vertical=True)
    fig.suptitle(f"{title}  —  MDF surowy 18 mm, {qty} szt.", fontsize=13, y=0.95)
    fig.text(0.06, 0.05, "\n".join(["Wymiary w mm. Układ odniesienia: lewy dolny róg (0, 0).", *notes]),
             fontsize=8, va="bottom")
    return fig, ax, pad


def field():
    fig, _, _ = page("Boisko", FIELD_L, FIELD_W, 1, ["Bez otworów."])
    return fig


def long_band():
    notes = [f"8 otworów przelotowych Ø{HOLE_D} (pod tuleje drążków), środki na wysokości {ROD_Y} od dolnej krawędzi.",
             "Środki otworów od lewej krawędzi: " + ", ".join(map(str, ROD_X)) + ".",
             "Obie bandy identyczne — otwory muszą się pokrywać po złożeniu."]
    fig, ax, pad = page("Banda długa", LONG_L, BAND_H, 2, notes)
    for x in ROD_X:
        ax.add_patch(Circle((x, ROD_Y), HOLE_D / 2, fill=False, lw=1))
        ax.plot([x, x], [ROD_Y - 20, -pad * 0.35], lw=0.3, color="grey", ls="--")
        ax.text(x, -pad * 0.4, str(x), ha="center", va="top", fontsize=7, rotation=90)
    dim(ax, LONG_L, 0, LONG_L, ROD_Y, f"{ROD_Y}", pad * 0.3, vertical=True)
    ax.text(ROD_X[0] + 20, ROD_Y + 18, f"Ø{HOLE_D}", fontsize=8)
    ax.plot([0, LONG_L], [FIELD_TOP, FIELD_TOP], lw=0.4, ls=":", color="grey")
    ax.text(LONG_L / 2, FIELD_TOP - 14, "poziom boiska (informacyjnie, bez obróbki)", ha="center", fontsize=7, color="grey")
    return fig


def short_band():
    x0 = (FIELD_W - GOAL_W) / 2
    notes = [f"Prostokątny otwór przelotowy {GOAL_W} × {GOAL_H}, wyśrodkowany: od x = {x0:g} do {x0 + GOAL_W:g},"
             f" od y = {FIELD_TOP} do {FIELD_TOP + GOAL_H}.",
             "Zaokrąglenie naroży wewnętrznych promieniem frezu (R ≤ 4) dopuszczalne."]
    fig, ax, pad = page("Banda krótka", FIELD_W, BAND_H, 2, notes)
    ax.add_patch(Rectangle((x0, FIELD_TOP), GOAL_W, GOAL_H, fill=False, lw=1))
    dim(ax, x0, FIELD_TOP + GOAL_H, x0 + GOAL_W, FIELD_TOP + GOAL_H, f"{GOAL_W}", 20)
    dim(ax, 0, 0, x0, 0, f"{x0:g}", -pad * 0.35)
    dim(ax, x0 + GOAL_W, FIELD_TOP, x0 + GOAL_W, FIELD_TOP + GOAL_H, f"{GOAL_H}", 20, vertical=True)
    dim(ax, FIELD_W, 0, FIELD_W, FIELD_TOP, f"{FIELD_TOP}", pad * 0.3, vertical=True)
    return fig


ROD_D = 16                  # rod diameter (standard steel tube)
HANDLE = 150                # rod sticking out on the handle side
ROD_OWNER_FRONT = [True, True, False, True, False, True, False, False]  # Left player's rods: handles on front side
RAIL = 20                   # 2020 profile
CROSSBARS_X = [T + 150, T + 450, T + 750, T + 1050]


def hidden(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.6, ls="--", color="grey"))


def assembly():
    """Three orthographic views of the assembled box: top, front (long side), side (short side)."""
    W = FIELD_W + 2 * T
    fig, ax = plt.subplots(figsize=(16.54, 11.69))  # A3 landscape
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle("Złożenie skrzyni — rzuty (mm)", fontsize=14, y=0.96)

    # --- top view, origin at (0, 0): x along table, y across (front side y = 0)
    oy = 0
    ax.add_patch(Rectangle((0, oy), LONG_L, W, fill=False, lw=1.5))
    ax.add_patch(Rectangle((T, oy + T), FIELD_L, FIELD_W, color="#d8ead8", lw=0))          # playfield
    for y in (oy, oy + W - T):
        ax.add_patch(Rectangle((0, y), LONG_L, T, color="#c9a77c", lw=0.8, ec="k"))         # long bands
    for x in (0, LONG_L - T):
        ax.add_patch(Rectangle((x, oy + T), T, FIELD_W, color="#e0c49c", lw=0.8, ec="k"))   # short bands
    gy = oy + T + (FIELD_W - GOAL_W) / 2
    for x in (0, LONG_L - T):
        ax.add_patch(Rectangle((x, gy), T, GOAL_W, color="white", lw=0.8, ec="k"))          # goal openings
    for x, front in zip(ROD_X, ROD_OWNER_FRONT):
        y0, y1 = (oy - HANDLE, oy + W + 30) if front else (oy - 30, oy + W + HANDLE)
        ax.add_patch(Rectangle((x - ROD_D / 2, y0), ROD_D, y1 - y0, color="#9aa", lw=0.5, ec="k"))
    for x in CROSSBARS_X:
        hidden(ax, x - RAIL / 2, oy + T + RAIL, RAIL, FIELD_W - 2 * RAIL)
    for y in (oy + T, oy + W - T - RAIL):
        hidden(ax, T, y, FIELD_L, RAIL)
    dim(ax, 0, oy + W + HANDLE, LONG_L, oy + W + HANDLE, f"{LONG_L}", 40)
    dim(ax, LONG_L, oy, LONG_L, oy + W, f"{W}", 60, vertical=True)
    ax.text(LONG_L / 2, oy - HANDLE - 40, "RZUT Z GÓRY  (dół rysunku = strona gracza L)", ha="center", fontsize=10)
    ax.text(-30, gy + GOAL_W / 2, f"bramka {GOAL_W}", rotation=90, ha="right", va="center", fontsize=8)

    # --- front view below the top view: x along table, z up (looking at the front long band)
    fz = oy - HANDLE - 120 - BAND_H
    ax.add_patch(Rectangle((0, fz), LONG_L, BAND_H, color="#c9a77c", lw=1.5, ec="k"))
    hidden(ax, T, fz + FIELD_TOP - T, FIELD_L, T)                                             # playfield
    hidden(ax, T, fz + FIELD_TOP - T - RAIL, FIELD_L, RAIL)                                   # rail
    for x in ROD_X:
        ax.add_patch(Circle((x, fz + ROD_Y), HOLE_D / 2, color="white", ec="k", lw=0.8))
        ax.add_patch(Circle((x, fz + ROD_Y), ROD_D / 2, color="#9aa", ec="k", lw=0.5))
    dim(ax, 0, fz, 0, fz + BAND_H, f"{BAND_H}", -50, vertical=True)
    dim(ax, LONG_L, fz, LONG_L, fz + FIELD_TOP, f"{FIELD_TOP}", 40, vertical=True)
    dim(ax, LONG_L, fz, LONG_L, fz + ROD_Y, f"{ROD_Y}", 110, vertical=True)
    ax.text(LONG_L / 2, fz - 50, f"RZUT Z PRZODU  (drążki co {ROD_PITCH}, oś {ROD_Z} nad boiskiem)", ha="center", fontsize=10)

    # --- side view to the right of the front view: y across, z up (looking at a short band)
    sx = LONG_L + 300
    ax.add_patch(Rectangle((sx, fz), W, BAND_H, color="#e0c49c", lw=1.5, ec="k"))
    for x in (sx, sx + W - T):
        ax.add_patch(Rectangle((x, fz), T, BAND_H, color="#c9a77c", lw=0.8, ec="k"))
    ax.add_patch(Rectangle((sx + T + (FIELD_W - GOAL_W) / 2, fz + FIELD_TOP), GOAL_W, GOAL_H,
                           color="white", lw=0.8, ec="k"))
    hidden(ax, sx + T, fz + FIELD_TOP - T, FIELD_W, T)
    ax.plot([sx - HANDLE, sx + W + HANDLE], [fz + ROD_Y] * 2, lw=2, color="#9aa")
    dim(ax, sx, fz + BAND_H, sx + W, fz + BAND_H, f"{W}", 40)
    dim(ax, sx + T + (FIELD_W - GOAL_W) / 2, fz, sx + T + (FIELD_W + GOAL_W) / 2, fz,
        f"bramka {GOAL_W} × {GOAL_H}", -60)
    ax.text(sx + W / 2, fz - 110, "RZUT Z BOKU", ha="center", fontsize=10)

    fig.text(0.05, 0.04, "Linie przerywane: elementy ukryte (blat, profile 2020 pod blatem). "
             f"Otwory w bandach Ø{HOLE_D} do potwierdzenia po wyborze tulei. Rączki drążków po stronie właściciela.",
             fontsize=9)
    ax.set_xlim(-250, sx + W + HANDLE + 100)
    ax.set_ylim(fz - 200, oy + W + HANDLE + 150)
    return fig


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {"zlozenie": assembly, "boisko": field, "banda-dluga": long_band, "banda-krotka": short_band}
    with PdfPages(out / "mdf-zamowienie.pdf") as pdf:
        for name, make in parts.items():
            fig = make()
            pdf.savefig(fig)
            fig.savefig(out / f"{name}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/foosball/drawings")
