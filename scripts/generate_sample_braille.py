"""Generate synthetic embossed-style Braille images for testing (no dataset required)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from braille.patterns import CHAR_TO_PATTERN  # noqa: E402

# Dot layout within one cell (px offsets from cell origin)
DOT_OFFSETS = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
}


def pattern_to_dots(pattern: int) -> list[int]:
    return [i + 1 for i in range(6) if pattern & (1 << i)]


def render_text(text: str, cell_px: int = 28, dot_r: int = 5, padding: int = 40) -> np.ndarray:
    cells = []
    for ch in text.lower():
        if ch == " ":
            cells.append(0)
        else:
            cells.append(CHAR_TO_PATTERN.get(ch, 0))

    cols = len(cells)
    w = padding * 2 + cols * cell_px * 2
    h = padding * 2 + cell_px * 4
    img = np.ones((h, w), dtype=np.uint8) * 245

    for ci, pattern in enumerate(cells):
        if pattern == 0:
            continue
        ox = padding + ci * cell_px * 2
        oy = padding
        for dot in pattern_to_dots(pattern):
            col, row = DOT_OFFSETS[dot]
            cx = int(ox + col * cell_px + cell_px // 2)
            cy = int(oy + row * cell_px + cell_px // 2)
            cv2.circle(img, (cx, cy), dot_r, 30, -1)
            cv2.circle(img, (cx, cy), dot_r, 0, 1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Generate sample Braille PNG images")
    parser.add_argument("--text", default="hello", help="English text to emboss")
    parser.add_argument("--out", default="samples", help="Output directory")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in args.text)
    path = out_dir / f"braille_{safe_name}.png"
    img = render_text(args.text)
    cv2.imwrite(str(path), img)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
