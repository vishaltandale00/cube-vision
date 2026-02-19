# Cube Vision v1 Issues to Fix

## Problem 1: Quad detection includes front face
The bottom edge of the detected quadrilateral extends too far down, capturing part of the front face (side stickers). This corrupts the bottom row of the 3x3 grid.

**Fix:** After finding the quad, verify that the warped image's bottom row actually contains cube-top-face colors (not side face). Alternatively, shrink the quad inward slightly (2-5%) to avoid edge bleeding.

## Problem 2: Blue stickers misclassified
Bottom row cells that are clearly blue (H~100-130) are being classified as yellow. The yellow HSV range (H=15-40, S>=35, V>=55) is too wide and may overlap with shadow-darkened yellows that shouldn't match.

**Fix:** Tighten yellow range. Ensure blue classification (H=95-130, S>=60, V>=50) runs BEFORE yellow in the priority chain, or add better disambiguation.

## Problem 3: OLL lookup table incomplete  
Only ~10 pattern entries exist. There are 57 OLL cases. Most inputs fall back to OLL 23.

**Fix:** Generate ALL 57 OLL cases as binary patterns. Each case maps to a specific yellow/not-yellow 9-bit pattern (with rotational equivalence since the user might hold the cube in any orientation). Build the complete lookup from references/oll.md descriptions.

## Problem 4: Orange/red confusion on stickerless cubes
Orange (H=8-24) and red (H=0-12) ranges overlap significantly. Stickerless cube orange can look very red-ish.

**Fix:** Use relative comparison — if both red and orange are detected, cluster them and assign based on which is more dominant in hue.

## Test images
- /Users/vishal/.openclaw/media/inbound/file_16---ba2ed61b-a72e-46fa-950b-7e43b8f0352b.jpg — Expected: cross + 2 unoriented corners (front)
- /Users/vishal/.openclaw/media/inbound/file_19---e47966cb-b0fc-48eb-8eaa-247e5bee723a.jpg — Expected: L-shape, top=[R,Y,Y,Y,Y,O,Y,B,B]
- /Users/vishal/.openclaw/media/inbound/file_20---816ee17c-b380-4131-adfa-05b1cdd8c045.jpg — Expected: same as file_19 (same cube state, same photo)
