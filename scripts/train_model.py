"""
Train Braille cell classifier on all practice datasets and save braille_classifier.pkl.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def discover_dataset_dirs() -> list[Path]:
    """Find all Braille letter datasets under project root."""
    found: list[Path] = []
    for p in sorted(ROOT.iterdir()):
        if not p.is_dir() or p.name in {
            "backend",
            "frontend",
            "web",
            "scripts",
            "docs",
            "samples",
            "models",
            ".git",
            ".github",
        }:
            continue
        name_lower = p.name.lower()
        if "braille" not in name_lower and "alphabet" not in name_lower:
            continue
        # Nested "Braille Dataset/Braille Dataset/*.jpg"
        inner = p / p.name
        if inner.is_dir() and any(inner.glob("*.jpg")):
            found.append(inner)
            continue
        if any(p.rglob("*.png")) or any(p.rglob("*.jpg")):
            found.append(p)
    if not found:
        raise FileNotFoundError(
            "No Braille datasets found. Add 'Braille Alphabet Image Dataset (A-Z)' and/or 'Braille Dataset'."
        )
    return found


def load_image(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot decode {path}")
    return img


def label_from_flat_filename(name: str) -> str | None:
    """e.g. a1.JPG0dim.jpg -> a"""
    m = re.match(r"^([a-zA-Z])", name)
    return m.group(1).upper() if m else None


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.4:
        angle = rng.uniform(-12, 12)
        h, w = out.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        h, w = out.shape[:2]
        scale = rng.uniform(0.85, 1.15)
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        out = cv2.resize(out, (nw, nh))
        out = cv2.resize(out, (w, h))
    if rng.random() < 0.3:
        noise = rng.normal(0, 8, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def preprocess_cell(img: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary.reshape(-1).astype(np.float32) / 255.0)


def iter_labeled_images(ds_dir: Path):
    """Yield (label, path) from folder-per-letter or flat JPG layouts."""
    subdirs = [d for d in ds_dir.iterdir() if d.is_dir()]
    letter_dirs = [d for d in subdirs if len(d.name) == 1 and d.name.isalpha()]

    if letter_dirs:
        for letter_dir in sorted(letter_dirs):
            label = letter_dir.name.upper()
            for fp in sorted(letter_dir.iterdir()):
                if fp.suffix.lower() in IMAGE_EXTS:
                    yield label, fp
        return

    for fp in sorted(ds_dir.iterdir()):
        if not fp.is_file() or fp.suffix.lower() not in IMAGE_EXTS:
            continue
        label = label_from_flat_filename(fp.name)
        if label:
            yield label, fp


def build_dataset(
    dataset_dirs: list[Path],
    augment_factor: int = 4,
    size: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    xs: list[np.ndarray] = []
    ys: list[str] = []
    sources: list[str] = []

    for ds_dir in dataset_dirs:
        count = 0
        for label, fp in iter_labeled_images(ds_dir):
            img = load_image(fp)
            xs.append(preprocess_cell(img, size))
            ys.append(label)
            count += 1
            for _ in range(augment_factor):
                aug = augment(
                    (img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                    rng,
                )
                xs.append(preprocess_cell(aug, size))
                ys.append(label)
        sources.append(f"{ds_dir.parent.name}/{ds_dir.name}: {count} images")
        print(f"  Loaded {count} images from {ds_dir}")

    return np.stack(xs), np.array(ys), sources


def train(
    dataset_dirs: list[Path],
    out_path: Path,
    augment_factor: int = 4,
) -> dict:
    print("Loading merged datasets:")
    x, y, sources = build_dataset(dataset_dirs, augment_factor=augment_factor)
    print(f"Total samples: {len(x)} (features={x.shape[1]})")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(1024, 512, 256),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=128,
                    learning_rate="adaptive",
                    max_iter=100,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=12,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("Training MLP classifier on merged data ...")
    clf.fit(x_train, y_train)

    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    print(f"Train accuracy: {acc_train:.4f}")
    print(f"Test accuracy:  {acc_test:.4f}")
    print(classification_report(y_test, pred_test, target_names=le.classes_))

    bundle = {
        "model": clf,
        "label_encoder": le,
        "image_size": 50,
        "version": "2.0.0",
        "classes": list(le.classes_),
        "metrics": {"train_accuracy": float(acc_train), "test_accuracy": float(acc_test)},
        "datasets": sources,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    print(f"Saved model to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", type=int, default=4, help="Augmented copies per image")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "braille_classifier.pkl",
    )
    args = parser.parse_args()
    dirs = discover_dataset_dirs()
    print("Dataset roots:")
    for d in dirs:
        print(f"  - {d}")
    train(dirs, args.out, augment_factor=args.augment)


if __name__ == "__main__":
    main()
