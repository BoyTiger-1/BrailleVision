"""Physical Braille dot detection and cell segmentation using OpenCV."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from braille.decoder import BrailleCell
from braille.decoder import cells_to_text
from braille.patterns import pattern_bits_from_dots


@dataclass
class Dot:
    x: float
    y: float
    radius: float


@dataclass
class DetectionResult:
    text: str
    cells: list[BrailleCell] = field(default_factory=list)
    dots: list[Dot] = field(default_factory=list)
    debug_image_b64: str | None = None
    alignment_hint: str = ""
    confidence: float = 0.0


def _preprocess(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return enhanced, blurred


def _binary_masks(blurred: np.ndarray) -> list[np.ndarray]:
    """Try dark-on-light and light-on-dark embossing."""
    masks = []
    for inv in (False, True):
        flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        _, otsu = cv2.threshold(blurred, 0, 255, flag + cv2.THRESH_OTSU)
        masks.append(otsu)
        adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, 31, 5
        )
        masks.append(adapt)
    return masks


def _find_dots(mask: np.ndarray, min_area: float, max_area: float) -> list[Dot]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots: list[Dot] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.35:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        dots.append(Dot(float(x), float(y), float(r)))
    return dots


def _pick_best_dots(gray: np.ndarray) -> list[Dot]:
    h, w = gray.shape[:2]
    img_area = h * w
    min_area = max(8, img_area * 0.00002)
    max_area = img_area * 0.02
    _, blurred = _preprocess(gray)
    best: list[Dot] = []
    best_score = -1.0
    for mask in _binary_masks(blurred):
        dots = _find_dots(mask, min_area, max_area)
        if len(dots) < 2:
            continue
        # Prefer plausible dot counts (not noise blobs)
        score = len(dots) * 10 - abs(len(dots) - 20) * 2
        if score > best_score:
            best_score = score
            best = dots
    return best


def _estimate_pitch(dots: list[Dot]) -> float:
    """Median nearest-neighbor distance ≈ spacing between adjacent dots."""
    if len(dots) < 2:
        return 20.0
    dists: list[float] = []
    for i, a in enumerate(dots):
        nearest = float("inf")
        for j, b in enumerate(dots):
            if i == j:
                continue
            d = math.hypot(a.x - b.x, a.y - b.y)
            if d < nearest:
                nearest = d
        if nearest < float("inf"):
            dists.append(nearest)
    return float(np.median(dists)) if dists else 20.0


def _estimate_cell_width(dots: list[Dot], pitch: float) -> float:
    """Horizontal period of one Braille character (≈ two dot columns)."""
    if len(dots) < 2:
        return pitch * 2.0
    xs = sorted(d.x for d in dots)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > pitch * 0.35]
    if not gaps:
        return pitch * 2.0
    gaps_sorted = sorted(gaps)
    # Bimodal: small gaps ≈ pitch (within cell), large gaps ≈ cell period
    mid = float(np.median(gaps_sorted))
    large = [g for g in gaps_sorted if g > mid * 1.25]
    if large:
        return float(np.median(large))
    return max(float(np.median(gaps_sorted)) * 2.0, pitch * 2.0)


def _assign_dot_to_position(dx: float, dy: float, cell_w: float, cell_h: float) -> int:
    """Map offset within cell bbox to dots 1-6."""
    col = 0 if dx < cell_w * 0.5 else 1
    third = cell_h / 3.0
    if dy < third:
        row = 0
    elif dy < 2 * third:
        row = 1
    else:
        row = 2
    table = [[1, 4], [2, 5], [3, 6]]
    return table[row][col]


def _group_dots_into_cells(dots: list[Dot]) -> list[list[Dot]]:
    """Group dots into 6-dot Braille cells (2D), not horizontal scan lines."""
    if not dots:
        return []
    pitch = _estimate_pitch(dots)
    cell_w = max(_estimate_cell_width(dots, pitch), pitch * 2.0)
    min_x = min(d.x for d in dots) - pitch * 0.5
    buckets: dict[int, list[Dot]] = {}
    for d in dots:
        idx = int((d.x - min_x) / cell_w)
        buckets.setdefault(idx, []).append(d)
    return [sorted(buckets[k], key=lambda d: d.y) for k in sorted(buckets)]


def _cell_groups_to_braille(cell_groups: list[list[Dot]], pitch: float) -> list[BrailleCell]:
    if not cell_groups:
        return []
    cells: list[BrailleCell] = []
    for col_idx, group in enumerate(cell_groups):
        min_x = min(d.x for d in group)
        min_y = min(d.y for d in group)
        cell_h = max(pitch * 3.0, (max(d.y for d in group) - min_y) + pitch * 0.5)
        cell_w = max(pitch * 1.8, (max(d.x for d in group) - min_x) + pitch * 0.5)
        active: list[int] = []
        for d in group:
            pos = _assign_dot_to_position(d.x - min_x, d.y - min_y, cell_w, cell_h)
            if pos not in active:
                active.append(pos)
        cells.append(BrailleCell(pattern=pattern_bits_from_dots(active), col=col_idx))
    return cells


def _alignment_hint(dot_count: int, cell_count: int) -> str:
    if dot_count == 0:
        return "No dots detected. Move closer, improve lighting, and center the Braille in frame."
    if dot_count < 3:
        return "Very few dots visible. Hold steady and fill more of the camera view."
    if cell_count == 0:
        return "Dots found but cells unclear. Align paper parallel to the screen edges."
    return "Good alignment. Hold steady for best results."


def detect_braille(image_bgr: np.ndarray, draw_debug: bool = True) -> DetectionResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dots = _pick_best_dots(gray)
    pitch = _estimate_pitch(dots)
    cell_groups = _group_dots_into_cells(dots)
    all_cells = _cell_groups_to_braille(cell_groups, pitch)

    # Multi-line: cluster cells by centroid Y when many cells wrap
    if len(all_cells) > 1 and dots:
        min_x = min(d.x for d in dots) - pitch
        cell_w = max(_estimate_cell_width(dots, pitch), pitch * 2.0)
        line_buckets: dict[int, list[BrailleCell]] = {}
        for c in all_cells:
            cx = min_x + (c.col + 0.5) * cell_w
            matching = [d for d in dots if abs(d.x - cx) < cell_w * 0.6]
            cy = float(np.mean([d.y for d in matching])) if matching else 0.0
            line_key = int(cy / (pitch * 4.5))
            line_buckets.setdefault(line_key, []).append(c)
        ordered: list[BrailleCell] = []
        for line_idx, key in enumerate(sorted(line_buckets)):
            line_cells = sorted(line_buckets[key], key=lambda c: c.col)
            for i, c in enumerate(line_cells):
                c.row = line_idx
                c.col = i
            ordered.extend(line_cells)
        all_cells = ordered

    text = cells_to_text(all_cells)
    conf = min(1.0, (len(dots) / max(len(all_cells) * 2, 1)) * 0.5 + 0.3) if all_cells else 0.1

    debug_b64 = None
    if draw_debug:
        vis = image_bgr.copy()
        for d in dots:
            cv2.circle(vis, (int(d.x), int(d.y)), int(max(d.radius, 4)), (0, 255, 0), 2)
        for cell in all_cells:
            cv2.putText(
                vis,
                hex(cell.pattern),
                (10 + cell.col * 30, 30 + cell.row * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 0),
                1,
            )
        import base64

        _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        debug_b64 = base64.b64encode(buf).decode("ascii")

    return DetectionResult(
        text=text,
        cells=all_cells,
        dots=dots,
        debug_image_b64=debug_b64,
        alignment_hint=_alignment_hint(len(dots), len(all_cells)),
        confidence=round(conf, 2),
    )
