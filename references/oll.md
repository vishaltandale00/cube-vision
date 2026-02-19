# OLL Algorithms (57 cases)

Standard Singmaster notation. Algorithms chosen for finger-trick friendliness.

## Dot Cases (no edges oriented)

| # | Name | Pattern | Algorithm | Recognition |
|---|------|---------|-----------|-------------|
| 1 | Dot 1 | No edges | R U2 R2 F R F' U2 R' F R F' | All 4 edges misoriented, no corners oriented |
| 2 | Dot 2 | No edges | F R U R' U' F' f R U R' U' f' | All 4 edges misoriented, 2 adjacent corners |
| 3 | Dot 3 | | f R U R' U' f' U' F R U R' U' F' | L-shape rotated with no edges |
| 4 | Dot 4 | | f R U R' U' f' U F R U R' U' F' | L-shape rotated with no edges (mirror) |

## Line Cases (2 opposite edges oriented)

| # | Name | Pattern | Algorithm | Recognition |
|---|------|---------|-----------|-------------|
| 5 | Line 1 | I-shape | r' U2 R U R' U r | Horizontal line + no corners |
| 6 | Line 2 | I-shape | r U2 R' U' R U' r' | Horizontal line + no corners (mirror) |
| 7 | Lightning | | r U R' U R U2 r' | Line + 1 corner top-left |
| 8 | Lightning M | | l' U' L U' L' U2 l | Mirror of 7 |
| 9 | Fish Line | | R U R' U' R' F R2 U R' U' F' | Line + corner pointing right |
| 10 | Fish Line M | | R U R' U R' F R F' R U2 R' | Line + corner pointing left |
| 11 | Slash | | r' R2 U R' U R U2 R' U M' | Line + 2 corners same side |
| 12 | Slash M | | M' R' U' R U' R' U2 R U' M | Mirror of 11 |
| 13 | Gun | | F U R U' R2 F' R U R U' R' | Line + L-shape corners |
| 14 | Gun M | | R' F R U R' F' R F U' F' | Mirror of 13 |
| 15 | Squish | | l' U' l L' U' L U l' U l | Line + opposite corners |
| 16 | Squish M | | r U r' R U R' U' r U' r' | Mirror of 15 |

## L-Shape Cases (2 adjacent edges oriented)

| # | Name | Pattern | Algorithm | Recognition |
|---|------|---------|-----------|-------------|
| 17 | L 1 | | R U R' U R' F R F' U2 R' F R F' | L-shape + no corners |
| 18 | L 2 | | r U R' U R U2 r2 U' R U' R' U2 r | L-shape + no corners (mirror) |
| 19 | L 3 | | r' R U R U R' U' M' R' F R F' | L + 1 corner back-right |
| 20 | L 4 | | r U R' U' M2 U R U' R' U' M' | L + 1 corner back-left |

## Cross Cases (all edges oriented)

### All Corners Oriented
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 21 | **SKIP** | (already solved) | Yellow cross + all corners yellow |

### One Corner Oriented
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 22 | Antisune | R U2 R' U' R U' R' | 1 corner oriented, pointing left |
| 23 | Sune | R U R' U R U2 R' | 1 corner oriented, pointing right |
| 24 | L 1 | r U R' U' r' F R F' | Cross + headlights back, 1 corner |
| 25 | L 2 | F' r U R' U' r' F R | Mirror of 24 |
| 26 | Antisune+ | R U2 R' U' R U R' U' R U' R' | Cross + no headlights, 1 corner |
| 27 | Sune+ | R U R' U R U' R' U R U2 R' | Mirror of 26 |

### Two Corners Oriented (adjacent)
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 28 | Steelworker | r U R' U' r' R U R U' R' | 2 adj corners, arrow shape |
| 29 | Awkward | R U R' U' R U' R' F' U' F R U R' | 2 adj corners, no headlights |
| 30 | Anti-awkward | F R' F R2 U' R' U' R U R' F2 | Mirror of 29 |
| 31 | Couch | R' U' F U R U' R' F' R | 2 adj corners, headlights right |
| 32 | Couch M | L U F' U' L' U L F L' | 2 adj corners, headlights left |

### Two Corners Oriented (diagonal)
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 33 | T shape | R U R' U' R' F R F' | Cross + T pattern (2 diagonal corners) |
| 34 | C shape | R U R2 U' R' F R U R U' F' | Cross + C pattern |
| 35 | Fish R | R U2 R2 F R F' R U2 R' | Fish pointing right |
| 36 | Fish L | L' U2 L2 F' L' F L' U2 L | Fish pointing left |
| 37 | Mounted Fish | F R' F' R U R U' R' | Fish shape, mounted |

### No Corners Oriented
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 38 | Mario | R U R' U R U' R' U' R' F R F' | Cross + no corners (W shape) |
| 39 | Awkward Fish | L F' L' U' L U F U' L' | Cross + 0 corners, bowtie |
| 40 | Awkward Fish M | R' F R U R' U' F' U R | Mirror of 39 |

### Two Corners Oriented (same side)
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 41 | Dalmatian | R U R' U R U2 R' F R U R' U' F' | Cross + 2 corners same side |
| 42 | Dalmatian M | R' U' R U' R' U2 R F R U R' U' F' | Mirror of 41 |
| 43 | P shape | F' U' L' U L F | Cross + P (headlights left, corner right) |
| 44 | P shape M | F U R U' R' F' | Cross + P mirror |
| 45 | T shape | F R U R' U' F' | **Your 2-look OLL!** Cross + T bar |
| 46 | S shape | R' U' R' F R F' U R | Cross + S/Z shape |
| 47 | S shape M | R' U' R' F R F' R' F R F' U R | Mirror of 46 |

### Four Corners Oriented (edges only wrong - shouldn't happen with cross)
| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 48 | Knight | F R U R' U' R U R' U' F' | Cross + knight move pattern |
| 49 | Double Bars | R B' R2 F R2 B R2 F' R | Parallel bars |
| 50 | Pinwheel | r' U' r U' R' U R U' R' U R r' U r | Cross + pinwheel |
| 51 | Bottlecap | F U R U' R' U R U' R' F' | Cross + antisune shape, bar front |
| 52 | Bottlecap M | R U R' U R U' B U' B' R' | Mirror of 51 |
| 53 | Frying Pan | r' U' R U' R' U R U' R' U2 r | Cross + frying pan |
| 54 | Frying Pan M | r U R' U R U' R' U R U2 r' | Mirror of 53 |
| 55 | Highway | R U2 R2 U' R U' R' U2 F R F' | Cross + highway |
| 56 | Highway M | r U r' U R U' R' U R U' R' r U' r' | Mirror of 55 |
| 57 | H perm OLL | R U R' U' M' U R U' r' | Cross + H pattern |

## Learning Priority (for sub-25)

### Phase 1: Cross OLLs you probably already know from 2-look
- 45 (F R U R' U' F'), 44, 43 — the bar/P shapes
- 22 (Antisune), 23 (Sune) — corner orientation

### Phase 2: High-frequency cross cases
- 33 (T shape), 37 (Mounted Fish), 35/36 (Fish)
- 28 (Steelworker), 31/32 (Couch)

### Phase 3: Remaining cross cases
- 24/25, 26/27, 34, 38, 39/40, 41/42, 46/47

### Phase 4: Dot and line cases (least common)
- Learn these last — they come up rarely and 2-look handles them fine
