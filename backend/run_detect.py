"""CLI: run ML + CV detection on an image file."""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from braille.ml_detector import detect_braille_ml


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_detect.py <image.png>")
        sys.exit(1)
    path = Path(sys.argv[1])
    img = cv2.imread(str(path))
    if img is None:
        raw = path.read_bytes()
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Cannot read {path}")
        sys.exit(1)
    result = detect_braille_ml(img)
    print(f"Text: {result.text!r}")
    print(f"Dots: {len(result.dots)}, Cells: {len(result.cells)}, Confidence: {result.confidence}")
    print(f"Hint: {result.alignment_hint}")
    if result.debug_image_b64:
        out = path.with_name(path.stem + "_debug.jpg")
        import base64

        out.write_bytes(base64.b64decode(result.debug_image_b64))
        print(f"Debug image: {out}")


if __name__ == "__main__":
    main()
