# Cube Vision — OLL/PLL Case Recognizer

## Goal
A Python CLI tool that takes a photo of a Rubik's cube and identifies the OLL or PLL case, returning the case number and algorithm.

## Usage
```bash
python cube_vision.py <image_path>              # JSON output
python cube_vision.py <image_path> --debug       # saves annotated debug images
python cube_vision.py <image_path> --mode oll    # force OLL detection
python cube_vision.py <image_path> --mode pll    # force PLL detection
```

## Output
```json
{
  "mode": "oll",
  "case": "OLL 12",
  "name": "Steelworker",
  "algorithm": "r U R' U' r' R U R U' R'",
  "confidence": 0.92,
  "detected_grid": {
    "top_face": ["yellow", "yellow", "yellow", "red", "yellow", "orange", "yellow", "blue", "blue"],
    "oriented": [true, true, true, false, true, false, true, false, false]
  }
}
```

## Detection Pipeline

### Step 1: Find the cube face
- Input: photo taken from roughly top-down angle (user holding phone above cube)
- The cube face is the dominant 3x3 colored grid in the image
- Approach: Convert to grayscale → adaptive threshold → find contours → filter for largest approximate quadrilateral
- Alternative: Use Hough line detection to find grid lines
- Must handle slight perspective (not perfectly top-down)

### Step 2: Perspective transform
- Once 4 corners of the face are found, warp to a perfect 300x300 square
- This normalizes the grid regardless of camera angle

### Step 3: Segment 9 cells
- Divide the 300x300 into 3x3 grid (100x100 each)
- Sample the CENTER 40% of each cell (avoid edges where colors bleed on stickerless cubes)
- Compute median HSV values for the sampled region

### Step 4: Classify colors
Use HSV thresholds (tuned for stickerless cubes under indoor lighting):

| Color  | H range      | S min | V min |
|--------|-------------|-------|-------|
| Yellow | 20-35       | 60    | 100   |
| Red    | 0-10, 170-180 | 80  | 80    |
| Orange | 10-22       | 80    | 80    |
| Blue   | 95-130      | 60    | 50    |
| Green  | 35-85       | 50    | 50    |
| White  | any         | <40   | 180+  |

Also implement k-means fallback: cluster the 9 cells into N clusters, assign colors based on cluster centroids. This handles lighting variation better.

### Step 5: Determine case

**For OLL:**
- Center is always the top color (yellow). Check which of the 8 remaining positions match.
- Edges: positions 1,3,5,7 (top, left, right, bottom in reading order)
- Corners: positions 0,2,6,8
- Count oriented edges → dot(0), L(2 adjacent), line(2 opposite), cross(4)
- Count oriented corners → 0, 1, 2-adj, 2-diag, 4
- Use lookup table mapping (edge_pattern, corner_pattern) → OLL case number
- NOTE: Some cases need side sticker info to disambiguate (e.g., which way a corner twists). For v1, return top candidates if ambiguous.

**For PLL:**
- All top stickers are the same color
- Read the top row of each visible side face (need at least 2-3 sides visible)
- Match the color pattern against 21 PLL cases
- This is harder from a single photo — may need side views

### Step 6: Load algorithms
- Read from ../references/oll.md and ../references/pll.md
- Parse the markdown tables to build a lookup dict

## Test Images
- /Users/vishal/.openclaw/media/inbound/file_16---ba2ed61b-a72e-46fa-950b-7e43b8f0352b.jpg (OLL case with 2 corners unoriented, all edges oriented)
- /Users/vishal/.openclaw/media/inbound/file_19---e47966cb-b0fc-48eb-8eaa-247e5bee723a.jpg (L-shape OLL)

## Dependencies
- opencv-python
- numpy
- (no ML frameworks needed — pure CV)

## Important Design Decisions
- Stickerless cubes have NO black borders between stickers — colors bleed at edges. MUST sample cell centers.
- Indoor lighting varies a LOT. The k-means fallback is essential.
- The cube won't always be perfectly centered or top-down. Perspective transform must be robust.
- When in doubt, return top 3 candidates with confidence scores rather than one wrong answer.

## File Structure
```
scripts/
  cube_vision.py          # Main CLI + library
  cube_colors.py          # Color classification module  
  cube_grid.py            # Grid detection + perspective transform
  cube_cases.py           # OLL/PLL case lookup tables
  SPEC.md                 # This file
```
