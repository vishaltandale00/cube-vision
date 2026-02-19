# Side Face Detection Spec (v3)

## Goal
Detect not just the top face but also the top row of 2 visible side faces from a single angled photo. This gives us 15 stickers total (9 top + 3 front + 3 right) — enough to fully identify any OLL case.

## Geometry
When a cube is photographed at a typical ~30-45° angle:
- Top face: visible as a quadrilateral (already detected)
- Front face: shares the BOTTOM EDGE of the top face quad, extends downward
- Right face: shares the RIGHT EDGE of the top face quad, extends rightward

```
        TL -------- TR
       / top face  / |
      /           /  |  ← right face (shares TR-BR edge)
    BL -------- BR   |
      |  front  |   /
      |  face   |  /
      FL ------ FR
```

## Detection Strategy

### Step 1: Find top face quad (existing)
We already have 4 corners: TL, TR, BL, BR

### Step 2: Identify which edges are "front" and "right"
- The bottom edge (BL→BR) is the top of the front face
- The right edge (TR→BR) is the top of the right face
- We need to find the 2 missing corners: FL (front-left) and FR (front-right, shared with right face bottom)

### Step 3: Detect side face boundaries
Option A: Use edge/line detection below BL-BR line and right of TR-BR line
Option B: Use color segmentation — side face stickers will be different from the dark desk background  
Option C: Extrapolate — the front face height ≈ top face side length. Project BL and BR downward by the same distance.

### Step 4: Perspective warp each side face
- Front face: warp the quadrilateral BL-BR-FR-FL to a rectangle
- Right face: warp TR-BR-FR-RR to a rectangle

### Step 5: Read top row only
- From front face warp: read top 1/3 → 3 stickers
- From right face warp: read top 1/3 → 3 stickers (or leftmost column depending on orientation)

## Output Format
```json
{
  "top_face": ["O","R","R","B","Y","G","G","O","G"],
  "front_row": ["Y","Y","Y"],
  "right_row": ["R","G","B"],
  "oll_case": "OLL 1",
  "algorithm": "R U2 R2 F R F' U2 R' F R F'"
}
```

## Test Images
All images in dataset/labels.json — most have visible front+right faces at angle.

## Important
- Not all photos show 2 side faces. Some are nearly top-down (file_22). Gracefully handle: if side faces not detected, fall back to top-face-only detection.
- The side face stickers may be partially occluded or distorted. Only read what's clearly visible.
- Side face colors help disambiguate OLL cases that share the same top pattern but differ in side sticker orientation.
