#!/usr/bin/env python3
"""Generate research pitch diagram for QD-Search.

Tells the research story: Problem → Approach → Results → Future.
Audience: potential funders. Emphasis on *why QD matters*.

v2: Fixed all text overflow. Tighter loop. Callouts inside panel.
    Algorithm descriptions shortened. Empty space filled.
"""

import os, math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Paths ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = os.environ.get("CANVAS_FONT_DIR", "")
OUTPUT = os.environ.get("PITCH_OUTPUT", str(REPO_ROOT / "docs/pitch/pitch.pdf"))

W, H = 1800, 1100

# ── Colors ───────────────────────────────────────────────────────
BG          = HexColor("#F8F5F0")
PANEL_BG    = HexColor("#EDEAE5")
PANEL_BD    = HexColor("#C8C4BC")
GRID_LINE   = HexColor("#EEEBE6")
TEAL        = HexColor("#17706E")
TEAL_LIGHT  = HexColor("#D5ECEA")
TEAL_BG     = HexColor("#E2EDEC")
CORAL       = HexColor("#C06848")
CORAL_LIGHT = HexColor("#F0DED5")
GOLD        = HexColor("#B08830")
GOLD_LIGHT  = HexColor("#F5ECD5")
TEXT_DARK   = HexColor("#252523")
TEXT_MID    = HexColor("#656058")
TEXT_LIGHT  = HexColor("#9E9890")
WHITE       = HexColor("#FFFFFF")
RED_LIGHT   = HexColor("#F2D5D0")
RED         = HexColor("#C05050")

# ── Fonts ────────────────────────────────────────────────────────
FM = {
    "title":  "BigShoulders-Bold",
    "head":   "BigShoulders-Bold",
    "body":   "InstrumentSans",
    "bodyB":  "InstrumentSans-Bold",
    "mono":   "GeistMono",
    "monoB":  "GeistMono-Bold",
    "acc":    "Jura-Light",
    "accM":   "Jura-Medium",
    "serif":  "CrimsonPro-Regular",
    "serifB": "CrimsonPro-Bold",
    "serifI": "CrimsonPro-Italic",
}
for alias, ttf in {
    "BigShoulders-Bold":   "BigShoulders-Bold.ttf",
    "InstrumentSans":      "InstrumentSans-Regular.ttf",
    "InstrumentSans-Bold": "InstrumentSans-Bold.ttf",
    "GeistMono":           "GeistMono-Regular.ttf",
    "GeistMono-Bold":      "GeistMono-Bold.ttf",
    "Jura-Light":          "Jura-Light.ttf",
    "Jura-Medium":         "Jura-Medium.ttf",
    "CrimsonPro-Regular":  "CrimsonPro-Regular.ttf",
    "CrimsonPro-Bold":     "CrimsonPro-Bold.ttf",
    "CrimsonPro-Italic":   "CrimsonPro-Italic.ttf",
}.items():
    pdfmetrics.registerFont(TTFont(alias, os.path.join(FONT_DIR, ttf)))


c = canvas.Canvas(OUTPUT, pagesize=(W, H))


# ══════════════════════════════════════════════════════════════════
#  Primitives
# ══════════════════════════════════════════════════════════════════

def txt(x, y, s, fk, sz, col=TEXT_DARK, anchor="l"):
    c.setFont(FM[fk], sz); c.setFillColor(col)
    {"l": c.drawString, "r": c.drawRightString, "c": c.drawCentredString}[anchor](x, y, s)

def rrect(x, y, w, h, r=5, fill=None, stroke=None, lw=0.6):
    if fill: c.setFillColor(fill)
    if stroke: c.setStrokeColor(stroke); c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, fill=bool(fill), stroke=bool(stroke))

def circ(cx, cy, r, fill=None, stroke=None, lw=0.5):
    if fill: c.setFillColor(fill)
    if stroke: c.setStrokeColor(stroke); c.setLineWidth(lw)
    c.circle(cx, cy, r, fill=bool(fill), stroke=bool(stroke))

def hl(x1, x2, y, col, lw=0.5):
    c.setStrokeColor(col); c.setLineWidth(lw); c.line(x1, y, x2, y)

def draw_mini_grid(gx, gy, cols, rows, cell_sz, occupied):
    gw, gh = cols * cell_sz, rows * cell_sz
    rrect(gx - 2, gy - 2, gw + 4, gh + 4, 3, fill=WHITE, stroke=PANEL_BD, lw=0.5)
    for row in range(rows):
        for col in range(cols):
            cx_c = gx + col * cell_sz
            cy_c = gy + (rows - 1 - row) * cell_sz
            if (row, col) in occupied:
                t = min(1, max(0, occupied[(row, col)]))
                rv = int(210 + (23 - 210) * t)
                gv = int(232 + (112 - 232) * t)
                bv = int(230 + (110 - 230) * t)
                c.setFillColor(HexColor(f"#{rv:02x}{gv:02x}{bv:02x}"))
                c.rect(cx_c + 1, cy_c + 1, cell_sz - 2, cell_sz - 2, fill=1, stroke=0)
            c.setStrokeColor(HexColor("#D0CDC6")); c.setLineWidth(0.3)
            c.rect(cx_c, cy_c, cell_sz, cell_sz, fill=0, stroke=1)
    return gw, gh

def wrap_text(x, y, text_str, fk, sz, col, max_w, leading=None):
    if leading is None: leading = sz * 1.35
    c.setFont(FM[fk], sz); c.setFillColor(col)
    words = text_str.split(); line = ""; cy = y
    for word in words:
        test = f"{line} {word}".strip()
        if c.stringWidth(test, FM[fk], sz) > max_w and line:
            c.drawString(x, cy, line); cy -= leading; line = word
        else:
            line = test
    if line: c.drawString(x, cy, line); cy -= leading
    return cy

def arrow_on_line(x1, y1, x2, y2, col=TEAL, sz=8):
    """Draw arrowhead at the midpoint of a line, pointing from (x1,y1) to (x2,y2)."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    angle = math.atan2(y2 - y1, x2 - x1)
    c.setFillColor(col); p = c.beginPath()
    p.moveTo(mx + sz * math.cos(angle),
             my + sz * math.sin(angle))
    p.lineTo(mx + sz * 0.7 * math.cos(angle + 2.6),
             my + sz * 0.7 * math.sin(angle + 2.6))
    p.lineTo(mx + sz * 0.7 * math.cos(angle - 2.6),
             my + sz * 0.7 * math.sin(angle - 2.6))
    p.close(); c.drawPath(p, fill=1, stroke=0)


# ══════════════════════════════════════════════════════════════════
#  Background
# ══════════════════════════════════════════════════════════════════
c.setFillColor(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
c.setStrokeColor(GRID_LINE); c.setLineWidth(0.12)
for gx in range(0, W + 1, 90): c.line(gx, 0, gx, H)
for gy in range(0, H + 1, 90): c.line(0, gy, W, gy)

# Top accent bar
c.setFillColor(TEAL); c.rect(0, H - 6, W, 6, fill=1, stroke=0)


# ══════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════
txt(55, H - 52, "QD-SEARCH", "title", 48, TEXT_DARK)
txt(55, H - 82, "Quality-Diversity Algorithms for Autonomous AI Research",
    "serifI", 17, TEXT_MID)
hl(55, W - 55, H - 96, PANEL_BD, 0.6)
txt(W - 55, H - 52, "UCSB NLP Lab", "accM", 11, TEXT_LIGHT, "r")
txt(W - 55, H - 68, "March 2026", "acc", 10, TEXT_LIGHT, "r")


# ══════════════════════════════════════════════════════════════════
#  Panel layout
# ══════════════════════════════════════════════════════════════════
PANEL_TOP = H - 115
PANEL_BOT = 95
PANEL_H   = PANEL_TOP - PANEL_BOT

P1_X, P1_W = 50,   380     # THE PROBLEM
P2_X, P2_W = 450,  760     # THE APPROACH
P3_X, P3_W = 1230, 520     # RESULTS & FUTURE

def draw_panel(x, w, title, subtitle=None, accent=TEAL):
    rrect(x + 2, PANEL_BOT - 2, w, PANEL_H, 8, fill=HexColor("#E0DDD5"))
    rrect(x, PANEL_BOT, w, PANEL_H, 8, fill=WHITE, stroke=PANEL_BD, lw=0.8)
    c.saveState()
    p = c.beginPath(); p.roundRect(x, PANEL_BOT, w, PANEL_H, 8)
    c.clipPath(p, stroke=0)
    c.setFillColor(HexColor("#F2F0EC")); c.rect(x, PANEL_TOP - 48, w, 48, fill=1, stroke=0)
    c.setFillColor(accent);              c.rect(x, PANEL_TOP - 3, w, 3, fill=1, stroke=0)
    c.restoreState()
    rrect(x, PANEL_BOT, w, PANEL_H, 8, stroke=PANEL_BD, lw=0.8)
    txt(x + 18, PANEL_TOP - 32, title, "head", 17, TEXT_DARK)
    if subtitle: txt(x + 18, PANEL_TOP - 47, subtitle, "body", 9, TEXT_LIGHT)

draw_panel(P1_X, P1_W, "THE PROBLEM", "Why diversity matters", RED)
draw_panel(P2_X, P2_W, "THE APPROACH", "Quality-Diversity Search Loop", TEAL)
draw_panel(P3_X, P3_W, "RESULTS & FUTURE", "Evidence + Roadmap", GOLD)


# ══════════════════════════════════════════════════════════════════
#  Panel 1: THE PROBLEM + QD Explanation
# ══════════════════════════════════════════════════════════════════
px, pw = P1_X + 22, P1_W - 44
y = PANEL_TOP - 68

y = wrap_text(px, y,
    "Standard LLM-driven research converges to dominant solutions. "
    "Agents find one good approach and exploit it — missing diverse "
    "high-quality alternatives.",
    "serif", 11.5, TEXT_MID, pw, leading=15)
y -= 10

# Evidence callout
rrect(px, y - 42, pw, 42, 4, fill=RED_LIGHT, stroke=RED, lw=0.6)
txt(px + 10, y - 15, "Feb 2026 study:", "bodyB", 9, RED)
txt(px + 10, y - 28, "Only 2/5 tasks showed genuine diversity", "body", 8.5, TEXT_DARK)
txt(px + 10, y - 40, "without explicit QD pressure", "body", 8.5, TEXT_DARK)
y -= 54

# ── QD Algorithm explanation ──
txt(px, y, "Quality-Diversity Algorithms", "bodyB", 10.5, TEAL)
y -= 16

y = wrap_text(px, y,
    "QD algorithms optimize for two objectives at once: solution quality "
    "(fitness) and behavioral diversity (coverage). Instead of returning "
    "a single best answer, they maintain an archive of high-performing "
    "solutions across different regions of the design space.",
    "serif", 10, TEXT_MID, pw, leading=13)
y -= 10

# MAP-Elites as example
rrect(px, y - 62, pw, 62, 4, fill=TEAL_BG, stroke=TEAL, lw=0.5)
txt(px + 10, y - 14, "MAP-Elites (example):", "bodyB", 9, TEAL)
wrap_text(px + 10, y - 28,
    "Divides the feature space into a grid of cells. Each cell stores "
    "only the best solution found for that niche. New solutions compete "
    "to enter the archive — improving quality while filling the map.",
    "body", 8, TEXT_MID, pw - 20, leading=11)
y -= 74

# Comparison grids (with investor-friendly axis labels)
txt(px + pw // 2, y, "Greedy vs. MAP-Elites", "bodyB", 10, TEXT_DARK, "c")
y -= 6
txt(px + pw // 2, y, "Each cell = a different approach niche", "body", 8, TEXT_LIGHT, "c")
y -= 16

cell_sz = 18
rows, cols = 5, 4
grid_h_px = rows * cell_sz

# Greedy: clustered in one region
greedy_cells = {(1, 1): 0.7, (1, 2): 0.8, (2, 1): 0.9, (2, 2): 0.75, (3, 1): 0.6}
gx1 = px + 18
gy1 = y - grid_h_px
draw_mini_grid(gx1, gy1, cols, rows, cell_sz, greedy_cells)
txt(gx1 + cols * cell_sz // 2, gy1 - 12, "Greedy Search", "bodyB", 8, RED, "c")
txt(gx1 + cols * cell_sz // 2, gy1 - 23, "5 cells / all clustered", "body", 7, TEXT_LIGHT, "c")

# QD: spread across the grid
qd_cells = {
    (0, 1): 0.7, (0, 3): 0.8, (1, 0): 0.6, (1, 2): 0.9,
    (2, 1): 0.75, (2, 2): 0.7, (2, 3): 0.6, (3, 0): 0.8,
    (4, 1): 0.7, (4, 3): 0.8,
}
gx2 = px + pw // 2 + 14
draw_mini_grid(gx2, gy1, cols, rows, cell_sz, qd_cells)
txt(gx2 + cols * cell_sz // 2, gy1 - 12, "MAP-Elites", "bodyB", 8, TEAL, "c")
txt(gx2 + cols * cell_sz // 2, gy1 - 23, "10 cells / spread out", "body", 7, TEXT_LIGHT, "c")

# VS label
txt(px + pw // 2, y - grid_h_px // 2 + 2, "vs", "serifI", 12, TEXT_LIGHT, "c")

y = gy1 - 30

# Key insight
rrect(px, y - 32, pw, 32, 4, fill=PANEL_BG, stroke=PANEL_BD, lw=0.4)
txt(px + pw // 2, y - 12, "Diversity is a hidden objective", "serifB", 10, TEXT_DARK, "c")
txt(px + pw // 2, y - 25, "in ML research — easy to miss, hard to recover",
    "serifI", 9, TEXT_MID, "c")


# ══════════════════════════════════════════════════════════════════
#  Panel 2: THE APPROACH — Harness + Composable QD Layer
# ══════════════════════════════════════════════════════════════════
px, pw = P2_X + 22, P2_W - 44
y = PANEL_TOP - 68

# Description
txt(px, y, "Three fixed stages form the research harness. The QD algorithm is a",
    "serif", 11.5, TEXT_MID)
y -= 16
txt(px, y, "composable, pluggable layer that decides what to explore next.",
    "serif", 11.5, TEXT_MID)
y -= 26

# ── Top row: three fixed stages (horizontal pipeline) ──
stage_w, stage_h = 150, 36
stage_gap = 50
total_stages_w = 3 * stage_w + 2 * stage_gap
stage_x0 = px + (pw - total_stages_w) // 2
stage_y = y - stage_h

stage_defs = [
    ("IDEATE",   "LLM proposes strategies"),
    ("EXECUTE",  "Parallel bwrap sandboxes"),
    ("EVALUATE", "Score + extract features"),
]

stage_centers = []
for i, (label, desc_text) in enumerate(stage_defs):
    sx = stage_x0 + i * (stage_w + stage_gap)
    stage_centers.append(sx + stage_w // 2)

    rrect(sx, stage_y, stage_w, stage_h, 5, fill=TEAL_BG, stroke=TEAL, lw=1.0)
    txt(sx + stage_w // 2, stage_y + stage_h // 2 - 2, label, "monoB", 10, TEAL, "c")
    # Description above
    txt(sx + stage_w // 2, stage_y + stage_h + 6, desc_text, "body", 8, TEXT_LIGHT, "c")

# Arrows between stages
for i in range(2):
    ax = stage_x0 + (i + 1) * (stage_w + stage_gap) - stage_gap // 2
    ay = stage_y + stage_h // 2
    c.setStrokeColor(TEAL); c.setLineWidth(2.0)
    c.line(stage_x0 + (i + 1) * stage_w + i * stage_gap + 4, ay,
           stage_x0 + (i + 1) * (stage_w + stage_gap) - 4, ay)
    arrow_on_line(stage_x0 + (i + 1) * stage_w + i * stage_gap, ay,
                  stage_x0 + (i + 1) * (stage_w + stage_gap), ay, TEAL, 9)

# "Research Harness" label
txt(stage_x0 - 4, stage_y + stage_h // 2 - 2, "HARNESS", "accM", 8, TEXT_LIGHT, "r")

y = stage_y - 22

# ── Feedback arrows: EVALUATE → QD box, QD box → IDEATE ──
qd_box_top = y
qd_box_h = 285
qd_box_y = qd_box_top - qd_box_h

# Down-arrow from EVALUATE to QD box
eval_cx = stage_centers[2]
c.setStrokeColor(TEAL); c.setLineWidth(2.0)
c.line(eval_cx, stage_y - 2, eval_cx, qd_box_top - 2)
# arrowhead pointing down
c.setFillColor(TEAL); p = c.beginPath()
p.moveTo(eval_cx, qd_box_top - 2)
p.lineTo(eval_cx - 5, qd_box_top + 6)
p.lineTo(eval_cx + 5, qd_box_top + 6)
p.close(); c.drawPath(p, fill=1, stroke=0)

# Up-arrow from QD box to IDEATE
ideate_cx = stage_centers[0]
c.setStrokeColor(CORAL); c.setLineWidth(2.0)
c.line(ideate_cx, qd_box_top - 2, ideate_cx, stage_y - 2)
# arrowhead pointing up
c.setFillColor(CORAL); p = c.beginPath()
p.moveTo(ideate_cx, stage_y - 2)
p.lineTo(ideate_cx - 5, stage_y - 10)
p.lineTo(ideate_cx + 5, stage_y - 10)
p.close(); c.drawPath(p, fill=1, stroke=0)

# Arrow labels
txt(eval_cx + 8, qd_box_top + 14, "results", "body", 7.5, TEAL)
txt(ideate_cx - 8, qd_box_top + 14, "parents +", "body", 7.5, CORAL, "r")
txt(ideate_cx - 8, qd_box_top + 3, "context", "body", 7.5, CORAL, "r")

# ── QD Algorithm Layer (big composable box) ──
qd_x = px + 8
qd_w = pw - 16

# Double border to emphasize composability
rrect(qd_x - 2, qd_box_y - 2, qd_w + 4, qd_box_h + 4, 8,
      fill=CORAL_LIGHT, stroke=CORAL, lw=0.4)
rrect(qd_x, qd_box_y, qd_w, qd_box_h, 7,
      fill=WHITE, stroke=CORAL, lw=1.2)

# Header
txt(qd_x + 14, qd_box_y + qd_box_h - 20,
    "QD ALGORITHM LAYER", "head", 15, CORAL)
txt(qd_x + qd_w - 14, qd_box_y + qd_box_h - 20,
    "composable · pluggable", "serifI", 9, CORAL, "r")

# Thin line under header
hl(qd_x + 14, qd_x + qd_w - 14, qd_box_y + qd_box_h - 28, CORAL_LIGHT, 0.5)

# --- Left side: Archive grid (with clear investor-friendly labels) ---
grid_area_x = qd_x + 18
grid_area_w = qd_w * 0.38

txt(grid_area_x, qd_box_y + qd_box_h - 46, "Feature Archive", "bodyB", 10, TEXT_DARK)
txt(grid_area_x, qd_box_y + qd_box_h - 60,
    "Each cell = one approach niche.", "body", 8, TEXT_LIGHT)
txt(grid_area_x, qd_box_y + qd_box_h - 72,
    "Only the best solution per cell is kept.", "body", 8, TEXT_LIGHT)

# Grid dimensions
GCOLS, GROWS = 4, 6
CW_C, CH_C = 24, 17
grid_w, grid_h = GCOLS * CW_C, GROWS * CH_C

ROW_LABEL_MARGIN = 38
grid_x = grid_area_x + ROW_LABEL_MARGIN

# Column labels ABOVE the grid — very generous spacing
col_labels_y = qd_box_y + qd_box_h - 100     # 28px below last description (-72)
grid_top = col_labels_y - 20                   # 20px below column label baselines
grid_y = grid_top - grid_h                     # grid extends downward

# Grid background (tight padding so it doesn't creep into label space)
rrect(grid_x - 2, grid_y - 2, grid_w + 4, grid_h + 4, 2,
      fill=HexColor("#FAFAF8"), stroke=PANEL_BD, lw=0.4)

occupied_main = {
    (0, 1): 0.7, (0, 3): 0.85, (1, 0): 0.6, (1, 2): 0.95,
    (2, 1): 0.75, (2, 2): 0.7, (2, 3): 0.6, (3, 0): 0.8,
    (4, 1): 0.7, (4, 3): 0.85, (5, 2): 0.65,
}
draw_mini_grid(grid_x, grid_y, GCOLS, GROWS, CW_C, occupied_main)

# Column labels ABOVE grid (drawn AFTER grid so they render on top)
for i, lbl in enumerate(["Modify", "Add", "Replace", "Hybrid"]):
    txt(grid_x + i * CW_C + CW_C // 2, col_labels_y, lbl, "body", 7, TEXT_MID, "c")

# Row labels (drawn AFTER grid so they render on top)
for i, lbl in enumerate(["Attn", "FFN", "Embed", "Norm", "Train", "Arch"]):
    txt(grid_x - 6, grid_y + (GROWS - 1 - i) * CH_C + CH_C // 2 - 3,
        lbl, "body", 7.5, TEXT_MID, "r")

# Coverage stat below grid
txt(grid_area_x + ROW_LABEL_MARGIN + grid_w // 2, grid_y - 14,
    f"{len(occupied_main)} / {GCOLS * GROWS} cells filled",
    "mono", 8, TEAL, "c")

# --- Right side: Composable algorithms stack ---
algo_x = qd_x + qd_w * 0.42
algo_w = qd_w * 0.55
algo_y = qd_box_y + qd_box_h - 46

txt(algo_x, algo_y, "Composable Algorithms", "bodyB", 9.5, TEXT_DARK)
txt(algo_x, algo_y - 14, "Swap any algorithm into the loop:", "body", 8, TEXT_LIGHT)
algo_y -= 30

# Algorithm entries with brief descriptions
algo_entries = [
    ("MAP-Elites",      TEAL,    "Keep the best solution in each grid cell"),
    ("CVT-MAP-Elites",  TEAL,    "Adaptive Voronoi regions instead of fixed grid"),
    ("CMA-ME",          GOLD,    "Learn which search directions improve the archive"),
    ("Novelty Search",  GOLD,    "Reward being different, regardless of fitness"),
    ("Curiosity-Driven",GOLD,    "Prioritize the most uncertain regions"),
    ("Go-Explore",      GOLD,    "Random exploration first, then optimize"),
]

for name, col, brief in algo_entries:
    bg = TEAL_BG if col == TEAL else GOLD_LIGHT
    rrect(algo_x, algo_y - 28, algo_w, 28, 3, fill=bg, stroke=col, lw=0.4)
    txt(algo_x + 8, algo_y - 11, name, "monoB", 8, col)
    txt(algo_x + 8, algo_y - 23, brief, "body", 7, TEXT_MID)
    algo_y -= 32

# Visual divider between archive and algorithm list
plug_x = qd_x + qd_w * 0.40 - 6
c.setStrokeColor(HexColor("#E8E5DF")); c.setLineWidth(1.5)
c.line(plug_x, qd_box_y + 20, plug_x, qd_box_y + qd_box_h - 32)

# ── Innovation callouts (below QD box, two columns) ──
y_inn = qd_box_y - 16
txt(px, y_inn, "Key Innovations", "bodyB", 10, TEAL)
y_inn -= 16

left_col, right_col = px, px + pw // 2 + 10
for yi_start, innovations in [
    (y_inn, [
        ("LLM feature classifier", "81% accuracy via Gemini"),
        ("Parallel bwrap sandboxes", "GPU isolation per executor"),
        ("Cell-targeted exploration", "Bias toward empty niches"),
    ]),
]:
    yi = yi_start
    for title, detail in innovations:
        circ(left_col + 4, yi + 3, 2.5, fill=TEAL)
        txt(left_col + 14, yi, title, "bodyB", 8.5, TEXT_DARK)
        txt(left_col + 14, yi - 12, detail, "body", 7.5, TEXT_LIGHT)
        yi -= 26

yi = y_inn
for title, detail in [
    ("Persistent ideator memory", "Context carries across turns"),
    ("Agent-agnostic harness", "OpenCode · Claude · Cursor · Codex"),
    ("Pluggable search strategy", "Greedy, MAP-Elites, or custom"),
]:
    circ(right_col + 4, yi + 3, 2.5, fill=TEAL)
    txt(right_col + 14, yi, title, "bodyB", 8.5, TEXT_DARK)
    txt(right_col + 14, yi - 12, detail, "body", 7.5, TEXT_LIGHT)
    yi -= 26

# ── Three QD metrics at very bottom ──
y_met = PANEL_BOT + 16
met_w = pw // 3 - 8
for i, (label, value, desc_text) in enumerate([
    ("Coverage", "23%", "niches explored"),
    ("QD Score", "10.5", "aggregate fitness"),
    ("Best Fitness", "0.951", "val_bpb (lower=better)"),
]):
    mx = px + i * (met_w + 12)
    rrect(mx, y_met, met_w, 44, 4, fill=TEAL_BG, stroke=TEAL, lw=0.5)
    txt(mx + met_w // 2, y_met + 27, value, "monoB", 13, TEAL, "c")
    txt(mx + met_w // 2, y_met + 14, label, "bodyB", 7.5, TEAL, "c")
    txt(mx + met_w // 2, y_met + 3, desc_text, "body", 6.5, TEXT_LIGHT, "c")


# ══════════════════════════════════════════════════════════════════
#  Panel 3: RESULTS & FUTURE
# ══════════════════════════════════════════════════════════════════
px, pw = P3_X + 22, P3_W - 44
y = PANEL_TOP - 68

txt(px, y, "NanoGPT Optimization (100 iterations)", "bodyB", 10, TEXT_DARK)
y -= 20

# Big metrics
half = pw // 2 - 6
rrect(px, y - 68, half, 68, 5, fill=TEAL_BG, stroke=TEAL, lw=0.6)
txt(px + 10, y - 16, "Best BPB", "body", 9, TEAL)
txt(px + 10, y - 46, "0.9507", "monoB", 24, TEAL)
txt(px + 10, y - 60, "3.3% below baseline", "body", 7.5, TEXT_LIGHT)

rx = px + half + 12
rrect(rx, y - 68, half, 68, 5, fill=CORAL_LIGHT, stroke=CORAL, lw=0.6)
txt(rx + 10, y - 16, "Coverage", "body", 9, CORAL)
txt(rx + 10, y - 46, "23%", "monoB", 24, CORAL)
txt(rx + 10, y - 60, "16 of 70 niches filled", "body", 7.5, TEXT_LIGHT)
y -= 84

# Discovered approaches
txt(px, y, "Discovered approaches:", "bodyB", 9, TEXT_DARK)
y -= 15
for d in ["Grouped Query Attention (GQA)",
          "SwiGLU gating mechanisms",
          "Deeper models (12–16 layers)",
          "Novel learning rate schedules",
          "Embedding modifications"]:
    circ(px + 5, y + 3, 2, fill=TEAL)
    txt(px + 14, y, d, "body", 8.5, TEXT_MID)
    y -= 13
y -= 8

# Divider
hl(px, px + pw, y, PANEL_BD, 0.5)
y -= 16

# Algorithm Roadmap
txt(px, y, "ALGORITHM ROADMAP", "head", 14, GOLD)
y -= 6

# Helper: draw algo entry with name + one-sentence description
def algo_entry(y, name, sentence, col, bg):
    rrect(px, y - 28, pw, 28, 3, fill=bg, stroke=col, lw=0.4)
    txt(px + 8, y - 11, name, "monoB", 8, col)
    txt(px + 8, y - 23, sentence, "body", 7, TEXT_MID)
    return y - 32

# Implemented
txt(px, y, "Implemented", "bodyB", 8.5, TEAL)
y -= 14
y = algo_entry(y, "MAP-Elites (Grid)",
    "Fixed bins — keeps the best solution found in each cell of a feature grid.",
    TEAL, TEAL_BG)
y = algo_entry(y, "CVT-MAP-Elites",
    "Adaptive Voronoi tessellation partitions continuous feature spaces naturally.",
    TEAL, TEAL_BG)
y = algo_entry(y, "LLM Feature Classifier",
    "Gemini classifies solutions into archive cells with 81% accuracy.",
    TEAL, TEAL_BG)
y -= 4

# Planned
txt(px, y, "In Pipeline", "bodyB", 8.5, GOLD)
y -= 14
y = algo_entry(y, "CMA-ME",
    "Learns which search directions in feature space produce the most archive gains.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Novelty Search",
    "Rewards behavioral novelty — solutions that differ most from everything seen so far.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Curiosity-Driven",
    "Prioritizes exploration where the model's predictions are most uncertain.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Novelty Checker",
    "Validates that proposed ideas are genuinely novel via archive embedding comparison.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Island Model",
    "Multiple independent populations that periodically exchange elite solutions.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Go-Explore",
    "Finds diverse states through random exploration, then optimizes the most promising.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Multi-Task Archive",
    "Single archive spanning domains — enabling cross-task knowledge transfer.",
    GOLD, GOLD_LIGHT)
y = algo_entry(y, "Dynamic Grids",
    "Auto-splits dense archive regions and merges sparse ones as coverage evolves.",
    GOLD, GOLD_LIGHT)


# ══════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════
hl(55, W - 55, 82, PANEL_BD, 0.5)

txt(55, 60, "Related:", "bodyB", 8.5, TEXT_LIGHT)
txt(105, 60, "CodeEvolve (arXiv:2510.14150)  ·  "
    "AIRA-Dojo (arXiv:2507.02554)  ·  "
    "AIDE (arXiv:2502.13138)  ·  "
    "OMNI-EPIC (arXiv:2405.15568)", "body", 8.5, TEXT_MID)

txt(55, 40, "Core thesis:", "bodyB", 8.5, TEXT_LIGHT)
wrap_text(120, 40,
    "Explicit diversity pressure yields qualitatively different — "
    "and often superior — solutions that greedy optimization misses.",
    "serifI", 9, TEXT_MID, W - 180)


# ── Save ─────────────────────────────────────────────────────────
c.save()
print(f"✓ {OUTPUT}")
