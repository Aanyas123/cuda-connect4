#!/usr/bin/env python3
"""Renders a game's moves.csv as an animated GIF plus a PNG of the final board.

Usage: render_game.py MOVES_CSV OUTPUT_PREFIX
Writes OUTPUT_PREFIX.gif and OUTPUT_PREFIX_final.png.
"""
import csv
import sys

from PIL import Image, ImageDraw

COLS, ROWS, CELL, TOP = 7, 6, 60, 50
COLORS = {1: (220, 40, 40), 2: (245, 200, 30)}


def draw(grid, title, last):
    img = Image.new("RGB", (COLS * CELL, ROWS * CELL + TOP), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 8), title, fill=(0, 0, 0))
    d.rectangle([0, TOP, COLS * CELL, TOP + ROWS * CELL], fill=(30, 70, 170))
    for c in range(COLS):
        for r in range(ROWS):
            x, y = c * CELL, TOP + (ROWS - 1 - r) * CELL
            fill = COLORS.get(grid[c][r], (255, 255, 255))
            d.ellipse([x + 6, y + 6, x + CELL - 6, y + CELL - 6], fill=fill)
            if last == (c, r):
                d.ellipse([x + 22, y + 22, x + CELL - 22, y + CELL - 22],
                          fill=(0, 0, 0))
    return img


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with open(sys.argv[1]) as f:
        moves = list(csv.DictReader(f))
    grid = [[0] * ROWS for _ in range(COLS)]
    names = {}
    frames = [draw(grid, "Start", None)]
    for m in moves:
        player, col = int(m["player"]), int(m["column"])
        names[player] = m["strategy"]
        row = grid[col].index(0)
        grid[col][row] = player
        title = (f"Move {m['move']}: P{player} ({m['strategy']}) -> col {col}"
                 f"  [{float(m['gpu_ms']):.1f} ms GPU]")
        frames.append(draw(grid, title, (col, row)))
    frames += [frames[-1]] * 6  # Hold the final position.
    frames[0].save(sys.argv[2] + ".gif", save_all=True,
                   append_images=frames[1:], duration=500, loop=0)
    frames[-1].save(sys.argv[2] + "_final.png")
    print(f"Wrote {sys.argv[2]}.gif ({len(moves)} moves, players {names})")


if __name__ == "__main__":
    main()
