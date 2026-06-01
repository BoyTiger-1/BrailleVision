# BrailleVision Hackathon 2026 — Submission Brief

## Project name

**BrailleVision** — Real-time physical Braille scanner

## Short explanation

BrailleVision uses a device camera to photograph **real embossed or handwritten Braille dots** on paper. OpenCV finds dot-shaped blobs, groups them into standard 6-dot cells, decodes Grade 1 English Braille, and displays plus speaks the result. The web interface is built for accessibility: high contrast, large controls, voice guidance, and camera alignment feedback.

## Technology stack

- Python, FastAPI, OpenCV, NumPy
- React, TypeScript, Vite
- Web Camera API, Web Speech API

## How physical Braille is detected

1. Frame converted to grayscale; CLAHE improves contrast.
2. Multiple threshold masks handle dark dots on light paper and embossed shadows.
3. Contours filtered by area and circularity → dot centroids.
4. Dots clustered into horizontal rows, then into cells using estimated cell width.
5. Each dot mapped to positions 1–6 in a 2×3 Braille cell → 6-bit pattern → English character.

## Demo video script (suggested)

1. Show problem: Unicode Braille vs physical paper.
2. Generate sample: `python scripts/generate_sample_braille.py --text hello`
3. Upload → recognized text + TTS.
4. Live webcam on printed/embossed sheet.
5. Show accessibility toggles and alignment hints.

## Accuracy / performance

- CPU-only, ~1 FPS live scan, sub-second single frames on laptop.
- Best results: even lighting, fill frame, hold parallel to edges.
- Synthetic samples: reliable; handwritten/embossed varies with print quality.

## Accessibility features

- High contrast theme
- Minimum 56px touch targets
- ARIA live region for recognized text
- Voice guidance on start and alignment issues
- Read aloud button + auto TTS
- Visual alignment frame overlay

## Future plan

- ML dot detector trained on real-world dataset
- Grade 2 Braille, mobile app, offline edge model

## Repository

GitHub: *(add your link after push)*

## Live demo

*(optional — deploy frontend + backend to Railway/Render/Fly)*
