# 🧊 Cube Vision

Computer vision OLL/PLL case recognizer for Rubik's cube. Snap a photo of your cube, get the case number + algorithm.

Built for speedcubers who want instant case identification from a phone photo — no smart cube required.

## How it works

1. **Detect** the cube face via contour detection + perspective transform
2. **Classify** each of the 9 sticker colors using HSV thresholds + k-means
3. **Identify** the OLL/PLL case from the color pattern
4. **Return** the case number, name, and algorithm (sourced from J Perm)

## Usage

```bash
python cube_vision.py photo.jpg                 # JSON output
python cube_vision.py photo.jpg --debug          # saves annotated debug images
python cube_vision.py photo.jpg --mode oll       # force OLL detection
```

## Output

```json
{
  "mode": "oll",
  "case": "OLL 12",
  "algorithm": "F (R U R' U') F' U F (R U R' U') F'",
  "confidence": 0.92,
  "top_face": ["Y", "Y", "Y", "R", "Y", "O", "Y", "B", "B"]
}
```

## Features

- Works with **stickerless cubes** (samples cell centers to avoid color bleed)
- Handles **varying lighting** via k-means color clustering fallback
- **Perspective correction** for non-top-down angles
- Returns **top 3 candidates** with confidence when ambiguous
- All 57 OLL cases + 21 PLL cases

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy
```

## Status

🚧 Under construction — built by [Rex](https://github.com/vishaltandale00) via OpenClaw coding agents.

## License

MIT
