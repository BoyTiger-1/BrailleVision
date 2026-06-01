"""Load trained .pkl and classify Braille cell crops."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

_MODEL_CACHE: dict | None = None
_DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "braille_classifier.pkl"


def load_model(path: Path | None = None) -> dict:
    global _MODEL_CACHE
    p = path or _DEFAULT_MODEL
    key = str(p.resolve())
    if _MODEL_CACHE is not None and _MODEL_CACHE.get("_path") == key:
        return _MODEL_CACHE
    if not p.is_file():
        raise FileNotFoundError(f"Model not found: {p}. Run: python scripts/train_model.py")
    bundle = joblib.load(p)
    bundle["_path"] = key
    _MODEL_CACHE = bundle
    return bundle


def preprocess_cell(bgr_or_gray: np.ndarray, size: int = 50) -> np.ndarray:
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    # Match training set: high-contrast binarized cell patches
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary.reshape(-1).astype(np.float32) / 255.0).reshape(1, -1)


def predict_cell(crop: np.ndarray, bundle: dict | None = None) -> tuple[str, float]:
    bundle = bundle or load_model()
    size = int(bundle.get("image_size", 50))
    x = preprocess_cell(crop, size=size)
    clf = bundle["model"]
    le: object = bundle["label_encoder"]
    proba = clf.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    letter = le.classes_[idx]  # type: ignore[attr-defined]
    return str(letter), float(proba[idx])


def predict_cells(crops: list[np.ndarray], bundle: dict | None = None) -> list[tuple[str, float]]:
    return [predict_cell(c, bundle) for c in crops]
