"""BrailleVision API — ML + CV physical Braille recognition."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from braille.classifier import load_model
from braille.ml_detector import detect_braille_ml

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="BrailleVision API",
    description="Physical Braille recognition using trained classifier (.pkl) + OpenCV segmentation.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


class ScanResponse(BaseModel):
    text: str
    confidence: float
    dot_count: int
    cell_count: int
    alignment_hint: str
    debug_image: str | None = None
    cells: list[dict[str, Any]]
    per_cell_confidence: list[float] = []


def _decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _result_payload(result, include_debug: bool = True) -> ScanResponse:
    per_cell = getattr(result, "per_cell_confidence", []) or []
    return ScanResponse(
        text=result.text or "",
        confidence=result.confidence,
        dot_count=len(result.dots),
        cell_count=len(result.cells),
        alignment_hint=result.alignment_hint,
        debug_image=result.debug_image_b64 if include_debug else None,
        cells=[
            {
                "pattern": c.pattern,
                "row": c.row,
                "col": c.col,
                "confidence": c.confidence,
            }
            for c in result.cells
        ],
        per_cell_confidence=per_cell,
    )


@app.on_event("startup")
def _warm_model():
    model_path = Path(__file__).resolve().parents[1] / "models" / "braille_classifier.pkl"
    if model_path.is_file():
        load_model(model_path)


@app.get("/health")
def health():
    model_path = Path(__file__).resolve().parents[1] / "models" / "braille_classifier.pkl"
    metrics = {}
    if model_path.is_file():
        try:
            b = load_model(model_path)
            metrics = b.get("metrics", {})
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "braillevision",
        "model_loaded": model_path.is_file(),
        "metrics": metrics,
    }


@app.get("/model/info")
def model_info():
    model_path = Path(__file__).resolve().parents[1] / "models" / "braille_classifier.pkl"
    if not model_path.is_file():
        return {"loaded": False, "error": "Model file missing. Run scripts/train_model.py"}
    b = load_model(model_path)
    return {
        "loaded": True,
        "version": b.get("version"),
        "classes": b.get("classes"),
        "image_size": b.get("image_size"),
        "metrics": b.get("metrics"),
        "dataset": b.get("dataset"),
    }


@app.post("/scan", response_model=ScanResponse)
async def scan_image(file: UploadFile = File(...)):
    data = await file.read()
    img = _decode_image_bytes(data)
    result = detect_braille_ml(img)
    return _result_payload(result)


class Base64ScanRequest(BaseModel):
    image: str
    include_debug: bool = True


@app.post("/scan/base64", response_model=ScanResponse)
async def scan_base64(body: Base64ScanRequest):
    raw = body.image
    if "," in raw:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    img = _decode_image_bytes(data)
    result = detect_braille_ml(img, draw_debug=body.include_debug)
    return _result_payload(result, include_debug=body.include_debug)


@app.post("/scan/batch", response_model=list[ScanResponse])
async def scan_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        data = await file.read()
        img = _decode_image_bytes(data)
        result = detect_braille_ml(img, draw_debug=False)
        results.append(_result_payload(result, include_debug=False))
    return results


class BatchBase64Request(BaseModel):
    images: list[str]
    include_debug: bool = False


@app.post("/scan/batch/base64", response_model=list[ScanResponse])
async def scan_batch_base64(body: BatchBase64Request):
    results = []
    for raw in body.images:
        img_raw = raw.split(",", 1)[1] if "," in raw else raw
        data = base64.b64decode(img_raw)
        img = _decode_image_bytes(data)
        result = detect_braille_ml(img, draw_debug=body.include_debug)
        results.append(_result_payload(result, include_debug=body.include_debug))
    return results


@app.websocket("/ws/scan")
async def ws_scan(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            payload = json.loads(msg)
            raw = payload.get("image", "")
            if "," in raw:
                raw = raw.split(",", 1)[1]
            data = base64.b64decode(raw)
            img = _decode_image_bytes(data)
            result = detect_braille_ml(img, draw_debug=payload.get("debug", False))
            await websocket.send_json(
                _result_payload(result, include_debug=payload.get("debug", False)).model_dump()
            )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
