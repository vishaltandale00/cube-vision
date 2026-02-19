# PLL Algorithms (21 cases)

All start with yellow on top. Algorithms optimized for finger tricks and speed.

## Edge-Only PLLs (4 cases)

| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 1 | **Ua** | R U' R U R U R U' R' U' R2 | 3-cycle edges CCW, headlights on left |
| 2 | **Ub** | R2 U R U R' U' R' U' R' U R' | 3-cycle edges CW, headlights on right |
| 3 | **H** | M2 U M2 U2 M2 U M2 | Swap opposite edge pairs, checkerboard sides |
| 4 | **Z** | M' U M2 U M2 U M' U2 M2 | Swap adjacent edge pairs, Z pattern on sides |

## Corner-Only PLLs (2 cases)

| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 5 | **Aa** | x R' U R' D2 R U' R' D2 R2 x' | 3-cycle corners CCW |
| 6 | **Ab** | x R2 D2 R U R' D2 R U' R x' | 3-cycle corners CW |

## Adjacent Corner Swap PLLs (8 cases)

| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 7 | **T** | R U R' U' R' F R2 U' R' U' R U R' F' | Headlights left + right, bar front |
| 8 | **F** | R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R | Headlights left, no bar |
| 9 | **Ja** | x R2 F R F' R U2 r' U r U2 x' | Headlights back + right, bar left |
| 10 | **Jb** | R U R' F' R U R' U' R' F R2 U' R' | Headlights left + back, bar right |
| 11 | **Ra** | R U' R' U' R U R D R' U' R D' R' U2 R' | Headlights back, 2x2 block right |
| 12 | **Rb** | R' U2 R U2 R' F R U R' U' R' F' R2 | Headlights right, 2x2 block left |
| 13 | **Ga** | R2 U R' U R' U' R U' R2 D U' R' U R D' | 3-corner + edge cycle |
| 14 | **Gb** | R' U' R U D' R2 U R' U R U' R U' R2 D | G-perm variant |
| 15 | **Gc** | R2 U' R U' R U R' U R2 D' U R U' R' D | G-perm variant |
| 16 | **Gd** | R U R' U' D R2 U' R U' R' U R' U R2 D' | G-perm variant |

## Diagonal Corner Swap PLLs (5 cases)

| # | Name | Algorithm | Recognition |
|---|------|-----------|-------------|
| 17 | **Y** | F R U' R' U' R U R' F' R U R' U' R' F R F' | No headlights on any side |
| 18 | **V** | R' U R' U' y R' F' R2 U' R' U R' F R F | No headlights, different pattern from Y |
| 19 | **Na** | R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R' | Diagonal swap + edge cycle, 1×3 bar |
| 20 | **Nb** | R' U R U' R' F' U' F R U R' F R' F' R U' R | Mirror of Na |
| 21 | **E** | x' R U' R' D R U R' D' R U R' D R U' R' D' x | No headlights on any side, all corners diagonal |

## Recognition Guide

### Step 1: Check for headlights (2 same-color stickers on a side)
- **4 sides with headlights** → H or Z (edge only)
- **2 opposite sides** → Ua, Ub, Aa, Ab
- **2 adjacent sides** → T, F, J, R, G perms
- **1 side (solved bar)** → Check which specific case
- **0 sides** → Y, V, N, E perms

### Step 2: Check for bars (3 same-color stickers in a row)
- Bars narrow it down further within each headlight group

## 2-Look → Full PLL Transition

### Cases you already know from 2-look:
- **Ua, Ub** (U-perms for edges)
- **Aa, Ab** (A-perms for corners)  
- **H, Z** (edge swaps)

### Learn next (most common, easy recognition):
1. **T** — very common, easy alg
2. **Jb** — shares pattern with T
3. **Y** — no headlights = Y or V, Y is more common
4. **Ra, Rb** — block recognition

### Learn last:
- **G perms** (4 of them, hardest to recognize)
- **N perms** (long algs, rare)
- **E perm** (rare, easy to confuse with Y/V)
