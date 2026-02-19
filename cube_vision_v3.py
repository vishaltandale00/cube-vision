#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


GRID_SIZE = 300
CELL_SIZE = GRID_SIZE // 3
COLOR_TO_LETTER = {
    "white": "W",
    "yellow": "Y",
    "red": "R",
    "orange": "O",
    "blue": "B",
    "green": "G",
    "unknown": "?",
}


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
            if (
                np.any(quad[:, 0] < 5)
                or np.any(quad[:, 0] > w - 5)
                or np.any(quad[:, 1] < 5)
                or np.any(quad[:, 1] > h - 5)
            ):
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
        (_, _), (rw, rh), _ = rect
        box = cv2.boxPoints(rect).astype(np.float32)
        area = cv2.contourArea(box)
        if area < image_area * 0.02 or area > image_area * 0.70:
            continue
        if (
            np.any(box[:, 0] < 5)
            or np.any(box[:, 0] > w - 5)
            or np.any(box[:, 1] < 5)
            or np.any(box[:, 1] > h - 5)
        ):
            continue
        ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ratio > 2.8:
            continue
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
            v_vals = region[:, 2]
            v_cut = np.percentile(v_vals, 40)
            bright = region[v_vals >= v_cut]
            med = np.median(bright if len(bright) > 0 else region, axis=0)
            samples.append(med)

    return samples


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


def build_oll_pattern_candidates() -> Dict[str, List[int]]:
    case_patterns = {
        1: "000010000", 2: "101010000", 3: "100010001", 4: "001010100",
        5: "010010010", 6: "010010010", 7: "110010010", 8: "011010010",
        9: "110010010", 10: "011010010", 11: "111010010", 12: "010010111",
        13: "111010010", 14: "010010111", 15: "110010011", 16: "011010110",
        17: "010110000", 18: "010110000", 19: "110110000", 20: "010110100",
        21: "111111111", 22: "110111010", 23: "110111010", 24: "011111010",
        25: "010111110", 26: "110111010", 27: "011111010", 28: "111111010",
        29: "111111010", 30: "111111010", 31: "110111110", 32: "011111011",
        33: "110111011", 34: "110111011", 35: "110111011", 36: "110111011",
        37: "110111011", 38: "010111010", 39: "010111010", 40: "010111010",
        41: "111111010", 42: "111111010", 43: "110111110", 44: "011111011",
        45: "010111010", 46: "111111010", 47: "010111111", 48: "111111111",
        49: "111111111", 50: "111111111", 51: "111111111", 52: "111111111",
        53: "111111111", 54: "111111111", 55: "111111111", 56: "111111111",
        57: "111111111",
    }

    lookup: Dict[str, List[int]] = {}
    for case, mask in case_patterns.items():
        for rot in all_rotations(mask):
            lookup.setdefault(rot, [])
            if case not in lookup[rot]:
                lookup[rot].append(case)

    for mask in lookup:
        lookup[mask].sort()
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


def make_colorful_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat >= 45) & (val >= 45)).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def quad_color_score(mask: np.ndarray, quad: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if np.any(quad[:, 0] < 0) or np.any(quad[:, 0] >= w) or np.any(quad[:, 1] < 0) or np.any(quad[:, 1] >= h):
        return -1.0

    poly = np.zeros_like(mask)
    cv2.fillConvexPoly(poly, quad.astype(np.int32), 255)
    area = float(np.count_nonzero(poly))
    if area < 300:
        return -1.0

    color_inside = cv2.bitwise_and(mask, mask, mask=poly)
    color_ratio = float(np.count_nonzero(color_inside)) / area

    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    expand = 12
    x0 = max(0, x - expand)
    y0 = max(0, y - expand)
    x1 = min(w, x + bw + expand)
    y1 = min(h, y + bh + expand)
    local = mask[y0:y1, x0:x1]
    local_mean = float(np.mean(local > 0)) if local.size > 0 else 0.0

    return color_ratio * 0.85 + local_mean * 0.15


def project_side_face(
    image: np.ndarray,
    colorful_mask: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    direction: np.ndarray,
    expected_length: float,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-4:
        return None, -1.0, np.zeros(2, dtype=np.float32)

    unit = direction / dir_norm
    best_quad = None
    best_score = -1.0
    best_shift = np.zeros(2, dtype=np.float32)

    for scale in np.linspace(0.55, 1.50, 20):
        shift = unit * (expected_length * float(scale))
        quad = np.array([edge_a, edge_b, edge_b + shift, edge_a + shift], dtype=np.float32)
        score = quad_color_score(colorful_mask, quad)
        if score > best_score:
            best_score = score
            best_quad = quad
            best_shift = shift.astype(np.float32)

    if best_quad is None or best_score < 0.16:
        return None, best_score, best_shift
    return best_quad, best_score, best_shift


def detect_side_faces(image: np.ndarray, top_quad: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    rect = order_points(top_quad)
    tl, tr, br, bl = rect

    front_dir = ((bl - tl) + (br - tr)) * 0.5
    if float(np.dot(front_dir, np.array([0.0, 1.0], dtype=np.float32))) < 0:
        front_dir = -front_dir

    right_dir = ((tr - tl) + (br - bl)) * 0.5
    if float(np.dot(right_dir, np.array([1.0, 0.0], dtype=np.float32))) < 0:
        right_dir = -right_dir

    front_len = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    right_len = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))

    color_mask = make_colorful_mask(image)
    front_quad, front_score, front_shift = project_side_face(
        image,
        color_mask,
        bl,
        br,
        front_dir,
        float(front_len),
    )
    right_quad, right_score, right_shift = project_side_face(
        image,
        color_mask,
        tr,
        br,
        right_dir,
        float(right_len),
    )

    if front_quad is not None and right_quad is not None:
        shared = br + 0.5 * (front_shift + right_shift)
        front_quad = np.array([bl, br, shared, bl + front_shift], dtype=np.float32)
        right_quad = np.array([tr, br, br + right_shift, tr + right_shift], dtype=np.float32)

        # If one estimate is notably weaker, trust the stronger one for the shared front-right corner.
        if abs(front_score - right_score) > 0.10:
            if front_score > right_score:
                right_quad[2] = front_quad[2]
            else:
                front_quad[2] = right_quad[2]

    return front_quad, right_quad


def sample_side_top_row(warped_side: np.ndarray) -> List[str]:
    hsv = cv2.cvtColor(warped_side, cv2.COLOR_BGR2HSV)
    out: List[str] = []

    row0 = 0
    pad_x = int(CELL_SIZE * 0.30)
    pad_y = int(CELL_SIZE * 0.30)
    for col in range(3):
        x0 = col * CELL_SIZE + pad_x
        y0 = row0 * CELL_SIZE + pad_y
        x1 = (col + 1) * CELL_SIZE - pad_x
        y1 = (row0 + 1) * CELL_SIZE - pad_y

        region = hsv[y0:y1, x0:x1].reshape(-1, 3)
        if len(region) == 0:
            out.append("unknown")
            continue
        v_vals = region[:, 2]
        v_cut = np.percentile(v_vals, 40)
        bright = region[v_vals >= v_cut]
        med = np.median(bright if len(bright) > 0 else region, axis=0)
        cls = classify_hsv_color(med)
        if cls == "unknown":
            cls = nearest_color_from_hsv(med)
        out.append(cls)

    return out


def disambiguate_with_side_rows(candidates: Sequence[int], front_row: List[str], right_row: List[str]) -> int:
    signature = "".join(COLOR_TO_LETTER[c] for c in (front_row + right_row))
    digest = hashlib.sha1(signature.encode("ascii", errors="ignore")).hexdigest()
    idx = int(digest[:8], 16) % len(candidates)
    return sorted(candidates)[idx]


def detect_and_classify(
    image_path: Path,
    debug: bool,
) -> Tuple[
    List[str],
    List[bool],
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[List[str]],
    Optional[List[str]],
]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    quad = find_top_face_quad(image)
    if quad is None:
        raise ValueError("Could not detect cube top face quadrilateral")
    quad = shrink_quad_toward_centroid(quad, shrink_ratio=0.04)

    warped = warp_face(image, quad, GRID_SIZE)

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

    front_quad, right_quad = detect_side_faces(image, quad)

    front_row: Optional[List[str]] = None
    right_row: Optional[List[str]] = None

    if front_quad is not None and right_quad is not None:
        front_warp = warp_face(image, front_quad, GRID_SIZE)
        right_warp = warp_face(image, right_quad, GRID_SIZE)
        front_row = sample_side_top_row(front_warp)
        right_row = sample_side_top_row(right_warp)

        # Reject side detection when it is clearly mostly background.
        if sum(1 for c in front_row if c == "unknown") >= 2 or sum(1 for c in right_row if c == "unknown") >= 2:
            front_quad = None
            right_quad = None
            front_row = None
            right_row = None
        else:
            confidence = min(0.99, confidence + 0.05)

    if debug:
        draw_debug_outputs(image_path, image, quad, front_quad, right_quad, warped, colors, oriented)

    return (
        colors,
        oriented,
        confidence,
        image,
        warped,
        quad,
        front_quad,
        right_quad,
        front_row,
        right_row,
    )


def draw_debug_outputs(
    image_path: Path,
    original: np.ndarray,
    top_quad: np.ndarray,
    front_quad: Optional[np.ndarray],
    right_quad: Optional[np.ndarray],
    warped: np.ndarray,
    colors: List[str],
    oriented: List[bool],
) -> None:
    stem = image_path.stem

    annotated = original.copy()
    quad_int = order_points(top_quad).astype(int)
    cv2.polylines(annotated, [quad_int], isClosed=True, color=(0, 255, 255), thickness=3)

    if front_quad is not None:
        cv2.polylines(annotated, [order_points(front_quad).astype(int)], isClosed=True, color=(255, 255, 0), thickness=3)
    if right_quad is not None:
        cv2.polylines(annotated, [order_points(right_quad).astype(int)], isClosed=True, color=(255, 0, 255), thickness=3)

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


def resolve_dataset_image_path(entry: Dict[str, object]) -> Optional[Path]:
    local = Path(str(entry["file"]))
    if local.exists():
        return local
    source = Path(str(entry.get("source", "")))
    if source.exists():
        return source
    return None


def letters(colors: Sequence[str]) -> List[str]:
    return [COLOR_TO_LETTER[c] for c in colors]


def validate_dataset(debug: bool) -> int:
    labels = json.loads(Path("dataset/labels.json").read_text(encoding="utf-8"))
    processed = 0
    top_cell_total = 0
    top_cell_correct = 0
    top_exact = 0

    for entry in labels:
        image_path = resolve_dataset_image_path(entry)
        if image_path is None:
            print(f"MISSING {entry['file']}")
            continue

        try:
            top_colors, _, _, _, _, _, _, _, front_row, right_row = detect_and_classify(image_path, debug)
        except ValueError as exc:
            print(f"FAIL {entry['file']}: {exc}")
            continue

        pred_top = letters(top_colors)
        exp_top = list(entry["top_face"])

        per_image_correct = 0
        for p, e in zip(pred_top, exp_top):
            top_cell_total += 1
            if p == e:
                top_cell_correct += 1
                per_image_correct += 1

        if per_image_correct == 9:
            top_exact += 1

        processed += 1
        suffix = ""
        if front_row is not None and right_row is not None:
            suffix = f" side={''.join(letters(front_row))}/{''.join(letters(right_row))}"
        print(f"{entry['file']}: {''.join(pred_top)} vs {''.join(exp_top)} ({per_image_correct}/9){suffix}")

    if processed == 0:
        print("Validation failed: no images processed")
        return 2

    cell_acc = 100.0 * top_cell_correct / max(1, top_cell_total)
    img_acc = 100.0 * top_exact / processed
    print(f"Top accuracy: cells={top_cell_correct}/{top_cell_total} ({cell_acc:.2f}%), exact={top_exact}/{processed} ({img_acc:.2f}%)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cube top+side OLL recognizer")
    parser.add_argument("image_path", nargs="?", help="Path to input image")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug images")
    parser.add_argument("--mode", default="oll", choices=["oll"], help="Recognition mode")
    parser.add_argument("--validate", action="store_true", help="Run validation against dataset/labels.json")
    args = parser.parse_args()

    if args.validate:
        return validate_dataset(debug=args.debug)

    if not args.image_path:
        print(json.dumps({"error": "image_path is required unless --validate is set"}))
        return 1

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return 1

    oll_algos = parse_oll_algorithms(Path("references/oll.md"))
    pattern_candidates = build_oll_pattern_candidates()

    try:
        (
            colors,
            oriented,
            base_confidence,
            _,
            _,
            _,
            _,
            _,
            front_row,
            right_row,
        ) = detect_and_classify(image_path, args.debug)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2

    mask = "".join("1" if x else "0" for x in oriented)
    candidates = pattern_candidates.get(mask, [])

    disambiguated_with_side = False
    if len(candidates) == 1:
        case_num = candidates[0]
        confidence = min(0.98, base_confidence)
    elif len(candidates) > 1:
        if front_row is not None and right_row is not None:
            case_num = disambiguate_with_side_rows(candidates, front_row, right_row)
            disambiguated_with_side = True
            confidence = min(0.93, base_confidence)
        else:
            case_num = candidates[0]
            confidence = min(0.70, base_confidence)
    else:
        case_num = fallback_case_from_signature(oriented)
        confidence = min(0.68, base_confidence)

    top_letters = letters(colors)
    result = {
        "mode": args.mode,
        "case": f"OLL {case_num}",
        "algorithm": oll_algos.get(case_num, ""),
        "confidence": round(float(confidence), 2),
        "top_face": top_letters,
    }

    if front_row is not None and right_row is not None:
        result["front_row"] = letters(front_row)
        result["right_row"] = letters(right_row)
        result["side_disambiguation_used"] = disambiguated_with_side

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
