#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


GRID_SIZE = 300
CELL_SIZE = GRID_SIZE // 3


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_face(image: np.ndarray, quad: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    rect = order_points(quad.astype("float32"))
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def shrink_quad_toward_centroid(quad: np.ndarray, shrink_ratio: float = 0.04) -> np.ndarray:
    center = np.mean(quad, axis=0, keepdims=True)
    return center + (quad - center) * (1.0 - shrink_ratio)


def find_top_face_quad(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    edges = cv2.Canny(blur, 50, 150)
    combined = cv2.bitwise_or(adapt, edges)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    h, w = image.shape[:2]
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_quad = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.05 or area > image_area * 0.85:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            if np.any(quad[:, 0] < 5) or np.any(quad[:, 0] > w - 5) or np.any(quad[:, 1] < 5) or np.any(quad[:, 1] > h - 5):
                continue

            sides = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                sides.append(float(np.linalg.norm(p2 - p1)))
            side_ratio = max(sides) / (min(sides) + 1e-6)
            if side_ratio > 2.8:
                continue

            center = np.mean(quad, axis=0)
            dist = float(np.linalg.norm(center - img_center))
            center_score = max(0.0, 1.0 - dist / math.sqrt(w * w + h * h))
            square_score = max(0.0, 1.0 - abs(1.0 - side_ratio))
            area_score = min(1.0, area / (image_area * 0.20))

            score = 0.70 * area_score + 0.20 * square_score + 0.10 * center_score
            if score > best_score:
                best_quad = quad
                best_score = score

    if best_quad is not None:
        return best_quad

    if not contours:
        return None

    fallback_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in fallback_contours:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        box = cv2.boxPoints(rect).astype(np.float32)
        area = cv2.contourArea(box)
        if area < image_area * 0.02 or area > image_area * 0.70:
            continue
        if np.any(box[:, 0] < 5) or np.any(box[:, 0] > w - 5) or np.any(box[:, 1] < 5) or np.any(box[:, 1] > h - 5):
            continue
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ratio > 2.8:
            continue
        _ = (cx, cy)  # kept for readability if scoring is extended later
        return box
    return None


def sample_cells_hsv(warped: np.ndarray) -> List[np.ndarray]:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    samples: List[np.ndarray] = []
    pad = int(CELL_SIZE * 0.30)

    for row in range(3):
        for col in range(3):
            x0 = col * CELL_SIZE + pad
            y0 = row * CELL_SIZE + pad
            x1 = (col + 1) * CELL_SIZE - pad
            y1 = (row + 1) * CELL_SIZE - pad

            region = hsv[y0:y1, x0:x1].reshape(-1, 3)
            # Bias sampling toward brighter pixels to reduce shadow effects.
            v_vals = region[:, 2]
            v_cut = np.percentile(v_vals, 40)
            bright = region[v_vals >= v_cut]
            med = np.median(bright if len(bright) > 0 else region, axis=0)
            samples.append(med)

    return samples


def sample_polygon_hsv(image: np.ndarray, polygon: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    poly = np.round(polygon).astype(np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    if cv2.contourArea(poly.astype(np.float32)) < 8.0:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    region = hsv[mask > 0]
    if len(region) == 0:
        return None

    v_vals = region[:, 2]
    v_cut = np.percentile(v_vals, 40)
    bright = region[v_vals >= v_cut]
    return np.median(bright if len(bright) > 0 else region, axis=0)


def edge_outward_normal(p0: np.ndarray, p1: np.ndarray, center: np.ndarray) -> np.ndarray:
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    n = np.array([-edge[1], edge[0]], dtype=np.float32) / length
    midpoint = (p0 + p1) * 0.5
    toward_center = center - midpoint
    if float(np.dot(n, toward_center)) > 0:
        n = -n
    return n


def sample_side_strip_row(
    image: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    seg_count: int = 3,
    d_min: float = 10.0,
    d_max: float = 20.0,
) -> Tuple[List[str], List[np.ndarray]]:
    n = edge_outward_normal(p0, p1, center)
    colors: List[str] = []
    polygons: List[np.ndarray] = []
    edge_vec = p1 - p0

    for i in range(seg_count):
        t0 = i / seg_count
        t1 = (i + 1) / seg_count
        a0 = p0 + edge_vec * t0 + n * d_min
        a1 = p0 + edge_vec * t1 + n * d_min
        b1 = p0 + edge_vec * t1 + n * d_max
        b0 = p0 + edge_vec * t0 + n * d_max
        poly = np.array([a0, a1, b1, b0], dtype=np.float32)
        polygons.append(poly)

        sample = sample_polygon_hsv(image, poly)
        if sample is None:
            colors.append("unknown")
        else:
            colors.append(classify_hsv_color(sample))

    return colors, polygons


def detect_side_strip_rows(
    image: np.ndarray, quad: np.ndarray
) -> Tuple[Dict[str, List[str]], Dict[str, List[np.ndarray]]]:
    q = order_points(quad.astype(np.float32))
    tl, tr, br, bl = q
    center = np.mean(q, axis=0)

    front_row, front_polys = sample_side_strip_row(image, bl, br, center)
    back_row, back_polys = sample_side_strip_row(image, tl, tr, center)
    left_row, left_polys = sample_side_strip_row(image, tl, bl, center)
    right_row, right_polys = sample_side_strip_row(image, tr, br, center)

    rows = {
        "front_row": front_row,
        "back_row": back_row,
        "left_row": left_row,
        "right_row": right_row,
    }
    polys = {
        "front_row": front_polys,
        "back_row": back_polys,
        "left_row": left_polys,
        "right_row": right_polys,
    }
    return rows, polys


def classify_hsv_color(hsv: np.ndarray) -> str:
    h, s, v = hsv

    if s < 28 and v >= 170:
        return "white"
    if 95 <= h <= 130 and s >= 60 and v >= 50:
        return "blue"
    if 22 <= h <= 35 and s >= 35 and v >= 55:
        return "yellow"
    if ((0 <= h <= 8) or (170 <= h <= 180)) and s >= 75 and v >= 60:
        return "red"
    if 8 < h <= 20 and s >= 70 and v >= 60:
        return "orange"
    if 35 < h < 85 and s >= 50 and v >= 50:
        return "green"
    return "unknown"


def nearest_color_from_hsv(hsv: np.ndarray) -> str:
    h, s, v = hsv
    if s < 35 and v > 170:
        return "white"

    anchors = {
        "red": 0,
        "orange": 16,
        "yellow": 27,
        "green": 60,
        "blue": 110,
    }

    def hue_dist(a: float, b: float) -> float:
        d = abs(a - b)
        return min(d, 180 - d)

    best = min(anchors.items(), key=lambda kv: hue_dist(h, kv[1]))[0]
    return best


def kmeans_fallback_colors(samples: List[np.ndarray], initial: List[str]) -> List[str]:
    if all(color != "unknown" for color in initial):
        return initial

    data = np.array(samples, dtype=np.float32)
    k = min(6, len(data))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )

    label_to_color: Dict[int, str] = {}
    for idx, center in enumerate(centers):
        direct = classify_hsv_color(center)
        label_to_color[idx] = direct if direct != "unknown" else nearest_color_from_hsv(center)

    out: List[str] = []
    for i, color in enumerate(initial):
        if color != "unknown":
            out.append(color)
        else:
            cluster = int(labels[i][0])
            out.append(label_to_color.get(cluster, "unknown"))
    return out


def rotate_mask(mask: str) -> str:
    idx = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    return "".join(mask[i] for i in idx)


def all_rotations(mask: str) -> List[str]:
    rots = [mask]
    for _ in range(3):
        rots.append(rotate_mask(rots[-1]))
    return rots


def build_oll_pattern_lookup() -> Dict[str, int]:
    # 1=yellow/oriented, 0=not-yellow. Center index 4 must be 1.
    # This table enumerates all 57 OLL cases from references/oll.md and maps each
    # case to a canonical top-face orientation mask inferred from the edge/corner
    # orientation description. Rotations are expanded below.
    case_patterns = {
        # Dot cases (no edges oriented)
        1: "000010000",   # 0 corners
        2: "101010000",   # 2 adjacent corners
        3: "100010001",   # 2 diagonal corners
        4: "001010100",   # 2 diagonal corners (mirror orientation)
        # Line cases (2 opposite edges oriented)
        5: "010010010",   # 0 corners
        6: "010010010",   # 0 corners (mirror)
        7: "110010010",   # 1 corner
        8: "011010010",   # 1 corner (mirror)
        9: "110010010",   # 1 corner
        10: "011010010",  # 1 corner (mirror)
        11: "111010010",  # 2 adjacent corners
        12: "010010111",  # 2 adjacent corners (mirror)
        13: "111010010",  # L-shape corners
        14: "010010111",  # L-shape corners (mirror)
        15: "110010011",  # 2 diagonal corners
        16: "011010110",  # 2 diagonal corners (mirror)
        # L-shape cases (2 adjacent edges oriented)
        17: "010110000",  # 0 corners
        18: "010110000",  # 0 corners (mirror)
        19: "110110000",  # 1 corner
        20: "010110100",  # 1 corner (mirror)
        # Cross cases (all edges oriented)
        21: "111111111",  # all corners oriented
        22: "110111010",  # 1 corner
        23: "110111010",  # 1 corner (mirror)
        24: "011111010",  # 1 corner
        25: "010111110",  # 1 corner (mirror)
        26: "110111010",  # 1 corner
        27: "011111010",  # 1 corner (mirror)
        28: "111111010",  # 2 adjacent corners
        29: "111111010",  # 2 adjacent corners
        30: "111111010",  # 2 adjacent corners (mirror)
        31: "110111110",  # 2 adjacent corners
        32: "011111011",  # 2 adjacent corners (mirror)
        33: "110111011",  # 2 diagonal corners
        34: "110111011",  # 2 diagonal corners
        35: "110111011",  # 2 diagonal corners
        36: "110111011",  # 2 diagonal corners (mirror)
        37: "110111011",  # 2 diagonal corners
        38: "010111010",  # 0 corners
        39: "010111010",  # 0 corners
        40: "010111010",  # 0 corners (mirror)
        41: "111111010",  # 2 corners same side
        42: "111111010",  # 2 corners same side (mirror)
        43: "110111110",  # P shape
        44: "011111011",  # P shape mirror
        45: "010111010",  # 2-look OLL T (cross + no corners)
        46: "111111010",  # S shape
        47: "010111111",  # S shape mirror
        48: "111111111",  # all corners oriented (per table heading)
        49: "111111111",  # all corners oriented
        50: "111111111",  # all corners oriented
        51: "111111111",  # all corners oriented
        52: "111111111",  # all corners oriented
        53: "111111111",  # all corners oriented
        54: "111111111",  # all corners oriented
        55: "111111111",  # all corners oriented
        56: "111111111",  # all corners oriented
        57: "111111111",  # all corners oriented
    }

    lookup: Dict[str, int] = {}
    for case in range(1, 58):
        mask = case_patterns[case]
        for rot in all_rotations(mask):
            lookup.setdefault(rot, case)
    return lookup


def pattern_signature(oriented: List[bool]) -> Tuple[int, int, str, str]:
    edges_idx = [1, 3, 5, 7]
    corners_idx = [0, 2, 6, 8]
    edges = [oriented[i] for i in edges_idx]
    corners = [oriented[i] for i in corners_idx]

    edge_count = int(sum(edges))
    corner_count = int(sum(corners))

    e_pat = "".join("1" if x else "0" for x in edges)
    c_pat = "".join("1" if x else "0" for x in corners)
    return edge_count, corner_count, e_pat, c_pat


def fallback_case_from_signature(oriented: List[bool]) -> int:
    edge_count, corner_count, _, _ = pattern_signature(oriented)

    if edge_count == 4 and corner_count == 4:
        return 21
    if edge_count == 4 and corner_count == 2:
        corner_idx = [0, 2, 6, 8]
        on = [i for i in corner_idx if oriented[i]]
        if len(on) == 2:
            if (on[0], on[1]) in {(0, 2), (2, 8), (8, 6), (6, 0)}:
                return 28
            return 33
    if edge_count == 2:
        edges = [oriented[i] for i in [1, 3, 5, 7]]
        if (edges[0] and edges[3]) or (edges[1] and edges[2]):
            return 5
        return 17
    if edge_count == 0:
        return 1
    return 23


def parse_oll_algorithms(path: Path) -> Dict[int, str]:
    text = path.read_text(encoding="utf-8")
    by_case: Dict[int, str] = {}
    algorithm_col: Optional[int] = None

    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if not parts:
            continue
        lowered = [p.lower() for p in parts]
        if "algorithm" in lowered:
            algorithm_col = lowered.index("algorithm")
            continue
        if len(parts) < 2 or algorithm_col is None or algorithm_col >= len(parts):
            continue
        if not re.fullmatch(r"\d+", parts[0]):
            continue
        case_num = int(parts[0])
        algorithm = parts[algorithm_col]
        if algorithm and algorithm != "(already solved)":
            by_case[case_num] = algorithm
        elif case_num == 21:
            by_case[case_num] = ""

    return by_case


def detect_and_classify(
    image_path: Path, debug: bool
) -> Tuple[List[str], List[bool], float, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    quad = find_top_face_quad(image)
    if quad is None:
        raise ValueError("Could not detect cube top face quadrilateral")
    quad = shrink_quad_toward_centroid(quad, shrink_ratio=0.04)

    warped = warp_face(image, quad, GRID_SIZE)
    side_rows, side_polygons = detect_side_strip_rows(image, quad)

    hsv_samples = sample_cells_hsv(warped)
    initial_colors = [classify_hsv_color(sample) for sample in hsv_samples]
    colors = kmeans_fallback_colors(hsv_samples, initial_colors)

    center_color = colors[4]
    top_color = "yellow" if center_color == "yellow" else center_color
    oriented = [c == top_color for c in colors]

    unknowns = sum(1 for c in colors if c == "unknown")
    confidence = max(0.25, 1.0 - unknowns * 0.12)
    if top_color != "yellow":
        confidence *= 0.75

    if debug:
        draw_debug_outputs(
            image_path,
            image,
            quad,
            warped,
            colors,
            oriented,
            side_rows,
            side_polygons,
        )

    return colors, oriented, confidence, image, warped, side_rows


def draw_debug_outputs(
    image_path: Path,
    original: np.ndarray,
    quad: np.ndarray,
    warped: np.ndarray,
    colors: List[str],
    oriented: List[bool],
    side_rows: Dict[str, List[str]],
    side_polygons: Dict[str, List[np.ndarray]],
) -> None:
    stem = image_path.stem

    annotated = original.copy()
    quad_int = order_points(quad).astype(int)
    cv2.polylines(annotated, [quad_int], isClosed=True, color=(0, 255, 255), thickness=3)
    for i, point in enumerate(quad_int):
        cv2.circle(annotated, tuple(point), 6, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            str(i),
            (int(point[0]) + 8, int(point[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for row_name, polys in side_polygons.items():
        row_colors = side_rows.get(row_name, [])
        for idx, poly in enumerate(polys):
            poly_i = np.round(poly).astype(np.int32)
            cv2.polylines(annotated, [poly_i], isClosed=True, color=(255, 0, 255), thickness=2)
            center = np.mean(poly, axis=0).astype(int)
            label = row_colors[idx][:1].upper() if idx < len(row_colors) else "?"
            cv2.putText(
                annotated,
                label,
                (int(center[0]) - 6, int(center[1]) + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    grid = warped.copy()
    for i in range(1, 3):
        cv2.line(grid, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), (255, 255, 255), 2)
        cv2.line(grid, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), (255, 255, 255), 2)

    color_bgr = {
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "white": (255, 255, 255),
        "unknown": (128, 128, 128),
    }

    for idx, color in enumerate(colors):
        row, col = divmod(idx, 3)
        cx = col * CELL_SIZE + CELL_SIZE // 2
        cy = row * CELL_SIZE + CELL_SIZE // 2
        bgr = color_bgr.get(color, (200, 200, 200))
        cv2.circle(grid, (cx, cy), 18, bgr, -1)
        tag = "Y" if oriented[idx] else "N"
        cv2.putText(
            grid,
            f"{idx}:{color[:1].upper()}/{tag}",
            (col * CELL_SIZE + 6, row * CELL_SIZE + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out1 = Path(f"{stem}_debug_detected.jpg")
    out2 = Path(f"{stem}_debug_grid.jpg")
    cv2.imwrite(str(out1), annotated)
    cv2.imwrite(str(out2), grid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cube top-face OLL recognizer")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug images")
    parser.add_argument("--mode", default="oll", choices=["oll"], help="Recognition mode")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return 1

    oll_algos = parse_oll_algorithms(Path("references/oll.md"))
    pattern_lookup = build_oll_pattern_lookup()

    try:
        colors, oriented, base_confidence, _, _, side_rows = detect_and_classify(image_path, args.debug)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    mask = "".join("1" if x else "0" for x in oriented)
    case_num = pattern_lookup.get(mask)
    if case_num is not None:
        confidence = min(0.98, base_confidence)
    else:
        case_num = fallback_case_from_signature(oriented)
        confidence = min(0.75, base_confidence)

    result = {
        "mode": args.mode,
        "case": f"OLL {case_num}",
        "algorithm": oll_algos.get(case_num, ""),
        "confidence": round(float(confidence), 2),
        "top_face": colors,
        "front_row": side_rows["front_row"],
        "back_row": side_rows["back_row"],
        "left_row": side_rows["left_row"],
        "right_row": side_rows["right_row"],
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
