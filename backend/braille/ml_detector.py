"""Full-image Braille recognition: CV segmentation + ML cell classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from braille.classifier import load_model, predict_cell
from braille.decoder import cells_to_text
from braille.detector import (
    Dot,
    DetectionResult,
    _cell_groups_to_braille,
    _estimate_pitch,
    _group_dots_into_cells,
    _pick_best_dots,
)


@dataclass
class MLDetectionResult(DetectionResult):
    per_cell_confidence: list[float] = field(default_factory=list)


def _crop_cell(image_bgr: np.ndarray, dots: list[Dot], pitch: float) -> np.ndarray:
    pad = int(pitch * 0.8)
    xs = [int(d.x) for d in dots]
    ys = [int(d.y) for d in dots]
    h, w = image_bgr.shape[:2]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(w, max(xs) + pad)
    y2 = min(h, max(ys) + pad)
    return image_bgr[y1:y2, x1:x2]


def _classify_whole_image(image_bgr: np.ndarray, bundle: dict) -> tuple[str, float]:
    letter, conf = predict_cell(image_bgr, bundle)
    return letter.lower(), conf


def detect_braille_ml(
    image_bgr: np.ndarray,
    draw_debug: bool = True,
    model_path=None,
) -> MLDetectionResult:
    bundle = load_model(model_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    dots = _pick_best_dots(gray)
    pitch = _estimate_pitch(dots) if dots else 20.0
    cell_groups = _group_dots_into_cells(dots) if len(dots) >= 2 else []

    from braille.decoder import BrailleCell

    confidences: list[float] = []
    cells: list[BrailleCell] = []
    text = ""

    # Single-cell practice images (~50x50) or one isolated cell in frame
    use_whole = (max(h, w) <= 80 and min(h, w) <= 80) or len(cell_groups) <= 1 and len(dots) < 8
    if use_whole or not cell_groups:
        text, conf = _classify_whole_image(image_bgr, bundle)
        confidences = [conf]
        cells = [BrailleCell(pattern=0, col=0, confidence=conf)]
    else:
        pattern_cells = _cell_groups_to_braille(cell_groups, pitch)
        # Multi-cell lines: geometric dot patterns (camera / embossed sheets)
        if len(cell_groups) >= 2:
            text = cells_to_text(pattern_cells)
            for col_idx, pc in enumerate(pattern_cells):
                crop = _crop_cell(image_bgr, cell_groups[col_idx], pitch)
                _, ml_conf = predict_cell(crop, bundle)
                confidences.append(ml_conf)
                cells.append(BrailleCell(pattern=pc.pattern, col=col_idx, confidence=ml_conf))
        else:
            crop = _crop_cell(image_bgr, cell_groups[0], pitch)
            ch, conf = predict_cell(crop, bundle)
            text = ch
            confidences = [conf]
            cells = [BrailleCell(pattern=0, col=0, confidence=conf)]
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    debug_b64 = None
    if draw_debug:
        vis = image_bgr.copy()
        for d in dots:
            cv2.circle(vis, (int(d.x), int(d.y)), int(max(d.radius, 4)), (0, 255, 0), 2)
        label_chars = list(text) if text else []
        for i, group in enumerate(cell_groups):
            if not group:
                continue
            ch = label_chars[i] if i < len(label_chars) else "?"
            x = int(min(d.x for d in group))
            y = int(min(d.y for d in group)) - 8
            cv2.putText(vis, f"{ch}", (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        import base64

        _, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        debug_b64 = base64.b64encode(buf).decode("ascii")

    hint = "Good alignment. Hold steady for best results."
    if len(dots) == 0:
        hint = "No dots detected. Move closer and improve lighting."
    elif len(cell_groups) == 0:
        hint = "Dots found but cells unclear. Align paper parallel to edges."

    return MLDetectionResult(
        text=text,
        cells=cells,
        dots=dots,
        debug_image_b64=debug_b64,
        alignment_hint=hint,
        confidence=round(avg_conf, 3),
        per_cell_confidence=confidences,
    )
