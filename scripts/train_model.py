"""
Train Braille cell classifier on the A-Z image dataset and save braille_classifier.pkl.
"""

from __future__ import annotations

import argparse
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


def find_dataset_dir() -> Path:
    for p in ROOT.iterdir():
        if p.is_dir() and "Alphabet" in p.name:
            return p
    raise FileNotFoundError("Braille Alphabet Image Dataset (A-Z) not found in project root")


def load_image(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot decode {path}")
    return img


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation for embossed-style robustness."""
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


def build_dataset(ds_dir: Path, augment_factor: int = 4, size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    xs: list[np.ndarray] = []
    ys: list[str] = []

    for letter_dir in sorted(ds_dir.iterdir()):
        if not letter_dir.is_dir():
            continue
        label = letter_dir.name.upper()
        if len(label) != 1 or not label.isalpha():
            continue
        files = sorted(letter_dir.glob("*.png"))
        for fp in files:
            img = load_image(fp)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            xs.append(img.reshape(-1).astype(np.float32) / 255.0)
            ys.append(label)
            for _ in range(augment_factor):
                aug = augment(img, rng)
                blur = cv2.GaussianBlur(aug, (3, 3), 0)
                _, aug = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                xs.append(aug.reshape(-1).astype(np.float32) / 255.0)
                ys.append(label)

    return np.stack(xs), np.array(ys)


def train(
    ds_dir: Path,
    out_path: Path,
    augment_factor: int = 4,
) -> dict:
    print(f"Loading dataset from {ds_dir} ...")
    x, y = build_dataset(ds_dir, augment_factor=augment_factor)
    print(f"Samples: {len(x)} (features={x.shape[1]})")

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
                    max_iter=80,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=12,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("Training MLP classifier ...")
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
        "version": "1.0.0",
        "classes": list(le.classes_),
        "metrics": {"train_accuracy": float(acc_train), "test_accuracy": float(acc_test)},
        "dataset": str(ds_dir.name),
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
    ds = find_dataset_dir()
    train(ds, args.out, augment_factor=args.augment)


if __name__ == "__main__":
    main()
