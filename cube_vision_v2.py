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
COLOR_ORDER = ["white", "yellow", "red", "orange", "blue", "green", "unknown"]
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


def hue_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 180.0 - d)


def classify_hsv_color(hsv: np.ndarray) -> str:
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    if s < 28 and v >= 170:
        return "white"
    if 95 <= h <= 130 and s >= 60 and v >= 50:
        return "blue"
    if 22 <= h <= 35 and s >= 35 and v >= 55:
        return "yellow"
    if ((0 <= h <= 8) or (170 <= h <= 180)) and s >= 75 and v >= 60:
        return "red"
    if 8 < h <= 21 and s >= 70 and v >= 60:
        return "orange"
    if 35 < h < 85 and s >= 50 and v >= 50:
        return "green"
    return "unknown"


def nearest_color_from_hsv(hsv: np.ndarray) -> str:
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    if s < 35 and v > 170:
        return "white"

    anchors = {
        "red": 0.0,
        "orange": 16.0,
        "yellow": 27.0,
        "green": 60.0,
        "blue": 110.0,
    }
    return min(anchors.items(), key=lambda kv: hue_distance(h, kv[1]))[0]


def top_face_plausibility(image: np.ndarray, quad: np.ndarray) -> float:
    preview = warp_face(image, shrink_quad_toward_centroid(quad, 0.05), size=120)
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    center_patch = hsv[50:70, 50:70].reshape(-1, 3)
    center_med = np.median(center_patch, axis=0)
    center_cls = classify_hsv_color(center_med)

    yellow_ratio = float(
        np.mean(
            (hsv[:, :, 0] >= 18)
            & (hsv[:, :, 0] <= 40)
            & (hsv[:, :, 1] >= 50)
            & (hsv[:, :, 2] >= 60)
        )
    )
    green_ratio = float(
        np.mean(
            (hsv[:, :, 0] >= 36)
            & (hsv[:, :, 0] <= 90)
            & (hsv[:, :, 1] >= 45)
            & (hsv[:, :, 2] >= 45)
        )
    )

    center = np.mean(quad, axis=0)
    upper_bias = 1.0 - float(center[1]) / max(1.0, image.shape[0])
    center_bonus = 1.0 if center_cls == "yellow" else 0.0
    return center_bonus * 0.65 + yellow_ratio * 0.95 + upper_bias * 0.10 - green_ratio * 0.45


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
    edges = cv2.Canny(blur, 40, 140)
    combined = cv2.bitwise_or(adapt, edges)

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    h, w = image.shape[:2]
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_quad = None
    best_score = float("-inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03 or area > image_area * 0.9:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        candidates: List[np.ndarray] = []
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx.reshape(4, 2).astype(np.float32))
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            candidates.append(box)

        for quad in candidates:
            if np.any(quad[:, 0] < 5) or np.any(quad[:, 0] > w - 5) or np.any(quad[:, 1] < 5) or np.any(quad[:, 1] > h - 5):
                continue

            sides = []
            for i in range(4):
                p1 = quad[i]
                p2 = quad[(i + 1) % 4]
                sides.append(float(np.linalg.norm(p2 - p1)))
            side_ratio = max(sides) / (min(sides) + 1e-6)
            if side_ratio > 4.2:
                continue

            center = np.mean(quad, axis=0)
            dist = float(np.linalg.norm(center - img_center))
            center_score = max(0.0, 1.0 - dist / math.sqrt(w * w + h * h))
            square_score = max(0.0, 1.0 - abs(1.0 - min(side_ratio, 2.2) / 2.2))
            area_score = min(1.0, area / (image_area * 0.18))

            score = 0.62 * area_score + 0.22 * square_score + 0.16 * center_score
            if score > best_score:
                best_quad = quad
                best_score = score

    return best_quad


def detect_quad_from_color_regions(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    image_area = h * w
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)

    best_quad = None
    best_score = float("-inf")

    for k in (4, 5, 6):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels_2d = labels.reshape(h, w)

        for idx in range(k):
            center = centers[idx]
            if center[1] < 130 and center[2] < 130:
                continue

            mask = (labels_2d == idx).astype(np.uint8) * 255
            mask = cv2.medianBlur(mask, 5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < image_area * 0.005:
                    continue
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect).astype(np.float32)
                box_area = cv2.contourArea(box)
                if box_area < image_area * 0.01 or box_area > image_area * 0.9:
                    continue

                (rw, rh) = rect[1]
                ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
                if ratio > 5.0:
                    continue

                plaus = top_face_plausibility(image, box)
                area_score = min(0.12, area / image_area)
                shape_score = max(0.0, 1.0 - abs(1.0 - min(ratio, 2.8) / 2.8))
                score = plaus + area_score * 0.15 + shape_score * 0.08
                if score > best_score:
                    best_score = score
                    best_quad = box

    return best_quad


def compute_cell_crops(warped: np.ndarray) -> List[np.ndarray]:
    cells = []
    for row in range(3):
        for col in range(3):
            x0 = col * CELL_SIZE
            y0 = row * CELL_SIZE
            x1 = (col + 1) * CELL_SIZE
            y1 = (row + 1) * CELL_SIZE
            cells.append(warped[y0:y1, x0:x1].copy())
    return cells


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
            v_cut = np.percentile(v_vals, 45)
            bright = region[v_vals >= v_cut]
            sample = np.median(bright if len(bright) > 0 else region, axis=0)
            samples.append(sample)

    return samples


def classify_by_hsv_thresholds(samples: List[np.ndarray]) -> List[str]:
    return [classify_hsv_color(s) for s in samples]


def classify_by_warp_kmeans(warped: np.ndarray) -> List[str]:
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    data = hsv.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    best_labels = None
    best_centers = None
    best_score = float("-inf")

    for k in (2, 3, 4):
        compactness, labels, centers = cv2.kmeans(
            data,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        score = -float(compactness) / max(1.0, len(data))
        if score > best_score:
            best_score = score
            best_labels = labels.reshape(GRID_SIZE, GRID_SIZE)
            best_centers = centers

    assert best_labels is not None and best_centers is not None

    cluster_color: Dict[int, str] = {}
    for idx, center in enumerate(best_centers):
        direct = classify_hsv_color(center)
        cluster_color[idx] = direct if direct != "unknown" else nearest_color_from_hsv(center)

    out: List[str] = []
    for row in range(3):
        for col in range(3):
            x0 = col * CELL_SIZE + int(CELL_SIZE * 0.2)
            y0 = row * CELL_SIZE + int(CELL_SIZE * 0.2)
            x1 = (col + 1) * CELL_SIZE - int(CELL_SIZE * 0.2)
            y1 = (row + 1) * CELL_SIZE - int(CELL_SIZE * 0.2)
            region = best_labels[y0:y1, x0:x1].reshape(-1)
            if len(region) == 0:
                out.append("unknown")
                continue
            values, counts = np.unique(region, return_counts=True)
            dominant_cluster = int(values[np.argmax(counts)])
            out.append(cluster_color.get(dominant_cluster, "unknown"))

    return out


def classify_by_cell_kmeans(samples: List[np.ndarray]) -> List[str]:
    data = np.array(samples, dtype=np.float32)
    if len(data) == 0:
        return []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    best_labels = None
    best_centers = None
    best_gap = -1.0

    prev_compactness = None
    chosen_k = 2
    for k in range(2, min(6, len(data)) + 1):
        compactness, labels, centers = cv2.kmeans(
            data,
            k,
            None,
            criteria,
            8,
            cv2.KMEANS_PP_CENTERS,
        )
        if prev_compactness is not None:
            gain = prev_compactness - compactness
            if gain > best_gap:
                best_gap = gain
                chosen_k = k
                best_labels = labels
                best_centers = centers
        else:
            best_labels = labels
            best_centers = centers
        prev_compactness = compactness

    if best_labels is None or best_centers is None:
        _, best_labels, best_centers = cv2.kmeans(
            data,
            chosen_k,
            None,
            criteria,
            8,
            cv2.KMEANS_PP_CENTERS,
        )

    label_to_color: Dict[int, str] = {}
    for idx, center in enumerate(best_centers):
        direct = classify_hsv_color(center)
        label_to_color[idx] = direct if direct != "unknown" else nearest_color_from_hsv(center)

    out: List[str] = []
    for i in range(len(samples)):
        cluster = int(best_labels[i][0])
        out.append(label_to_color.get(cluster, "unknown"))
    return out


def vote_cell_colors(votes: List[List[str]]) -> List[str]:
    if not votes:
        return []

    cells = len(votes[0])
    out: List[str] = []
    for i in range(cells):
        counts: Dict[str, int] = {}
        # Weight HSV classifier higher when it returns a known color.
        hsv_color = votes[0][i]
        hsv_weight = 2 if hsv_color != "unknown" else 1
        counts[hsv_color] = counts.get(hsv_color, 0) + hsv_weight
        for classifier in votes[1:]:
            color = classifier[i]
            counts[color] = counts.get(color, 0) + 1
        max_count = max(counts.values())
        top_colors = [c for c, n in counts.items() if n == max_count]
        if hsv_color in top_colors:
            color = hsv_color
        else:
            top_colors.sort(key=lambda c: COLOR_ORDER.index(c) if c in COLOR_ORDER else 999)
            color = top_colors[0]
        if color == "unknown" and len(counts) > 1:
            ordered = sorted(
                counts.items(),
                key=lambda kv: (-kv[1], COLOR_ORDER.index(kv[0]) if kv[0] in COLOR_ORDER else 999),
            )
            color = ordered[1][0]
        out.append(color)
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

    lookup: Dict[str, int] = {}
    for case in range(1, 58):
        for rot in all_rotations(case_patterns[case]):
            lookup.setdefault(rot, case)
    return lookup


def pattern_signature(oriented: List[bool]) -> Tuple[int, int, str, str]:
    edges_idx = [1, 3, 5, 7]
    corners_idx = [0, 2, 6, 8]
    edges = [oriented[i] for i in edges_idx]
    corners = [oriented[i] for i in corners_idx]
    return int(sum(edges)), int(sum(corners)), "".join("1" if x else "0" for x in edges), "".join("1" if x else "0" for x in corners)


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


def save_cell_crops(cells: List[np.ndarray], image_path: Path) -> Path:
    out_dir = Path("dataset") / "crops" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, cell in enumerate(cells):
        cv2.imwrite(str(out_dir / f"cell_{idx}.jpg"), cell)
    return out_dir


def save_grid_image(warped: np.ndarray, colors: List[str], image_path: Path) -> Path:
    grid = warped.copy()
    for i in range(1, 3):
        cv2.line(grid, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), (255, 255, 255), 2)
        cv2.line(grid, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), (255, 255, 255), 2)

    for idx, color in enumerate(colors):
        row, col = divmod(idx, 3)
        cv2.putText(
            grid,
            f"{idx}:{COLOR_TO_LETTER.get(color, '?')}",
            (col * CELL_SIZE + 8, row * CELL_SIZE + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    out_path = Path(f"{image_path.stem}_grid_summary.jpg")
    cv2.imwrite(str(out_path), grid)
    return out_path


def draw_debug_outputs(
    image_path: Path,
    original: np.ndarray,
    quad: np.ndarray,
    warped: np.ndarray,
    colors: List[str],
    oriented: List[bool],
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

    grid = warped.copy()
    for i in range(1, 3):
        cv2.line(grid, (0, i * CELL_SIZE), (GRID_SIZE, i * CELL_SIZE), (255, 255, 255), 2)
        cv2.line(grid, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE), (255, 255, 255), 2)

    for idx, color in enumerate(colors):
        row, col = divmod(idx, 3)
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

    cv2.imwrite(str(Path(f"{stem}_debug_detected.jpg")), annotated)
    cv2.imwrite(str(Path(f"{stem}_debug_grid.jpg")), grid)


def detect_and_classify(
    image_path: Path,
    debug: bool,
    save_crops: bool,
    save_grid: bool,
) -> Tuple[List[str], List[bool], float, np.ndarray, np.ndarray, Dict[str, str]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    quad = find_top_face_quad(image)
    color_quad = detect_quad_from_color_regions(image)

    if quad is None and color_quad is None:
        raise ValueError("Could not detect cube top face quadrilateral")

    method = "quad"
    if quad is None:
        quad = color_quad
        method = "color-regions"
    elif color_quad is not None:
        quad_score = top_face_plausibility(image, quad)
        color_score = top_face_plausibility(image, color_quad)
        # If geometric quad looks implausible as a top face, prefer color-region fallback.
        if quad_score < 0.42 and color_score > quad_score:
            quad = color_quad
            method = "color-regions"

    assert quad is not None
    quad = shrink_quad_toward_centroid(quad, shrink_ratio=0.05)
    warped = warp_face(image, quad, GRID_SIZE)

    samples = sample_cells_hsv(warped)
    hsv_votes = classify_by_hsv_thresholds(samples)
    warp_votes = classify_by_warp_kmeans(warped)
    cell_votes = classify_by_cell_kmeans(samples)
    colors = vote_cell_colors([hsv_votes, warp_votes, cell_votes])

    center_color = colors[4]
    top_color = "yellow" if center_color == "yellow" else center_color
    oriented = [c == top_color for c in colors]

    unknowns = sum(1 for c in colors if c == "unknown")
    confidence = max(0.30, 1.0 - unknowns * 0.12)
    if method == "color-regions":
        confidence *= 0.9
    if top_color != "yellow":
        confidence *= 0.75

    artifacts: Dict[str, str] = {}
    cells = compute_cell_crops(warped)
    if save_crops:
        artifacts["crops_dir"] = str(save_cell_crops(cells, image_path))
    if save_grid:
        artifacts["grid_image"] = str(save_grid_image(warped, colors, image_path))

    if debug:
        draw_debug_outputs(image_path, image, quad, warped, colors, oriented)

    return colors, oriented, confidence, image, warped, artifacts


def resolve_dataset_image_path(entry: Dict[str, object]) -> Optional[Path]:
    local = Path(str(entry["file"]))
    if local.exists():
        return local
    source = Path(str(entry.get("source", "")))
    if source.exists():
        return source
    return None


def validate_dataset(debug: bool, save_crops: bool, save_grid: bool) -> int:
    labels = json.loads(Path("dataset/labels.json").read_text(encoding="utf-8"))
    total_cells = 0
    correct_cells = 0
    image_exact = 0
    processed = 0

    for entry in labels:
        image_path = resolve_dataset_image_path(entry)
        if image_path is None:
            print(f"MISSING {entry['file']}")
            continue

        try:
            colors, _, _, _, _, _ = detect_and_classify(
                image_path,
                debug=debug,
                save_crops=save_crops,
                save_grid=save_grid,
            )
        except ValueError as exc:
            print(f"FAIL {entry['file']}: {exc}")
            continue

        predicted = [COLOR_TO_LETTER[c] for c in colors]
        expected = list(entry["top_face"])

        processed += 1
        per_image_correct = 0
        for p, e in zip(predicted, expected):
            total_cells += 1
            if p == e:
                correct_cells += 1
                per_image_correct += 1

        if per_image_correct == 9:
            image_exact += 1

        print(f"{entry['file']}: {''.join(predicted)} vs {''.join(expected)} ({per_image_correct}/9)")

    if processed == 0:
        print("Validation failed: no images processed")
        return 2

    cell_acc = 100.0 * correct_cells / max(1, total_cells)
    img_acc = 100.0 * image_exact / processed
    print(f"Accuracy: cells={correct_cells}/{total_cells} ({cell_acc:.2f}%), exact={image_exact}/{processed} ({img_acc:.2f}%)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Cube top-face OLL recognizer")
    parser.add_argument("image_path", nargs="?", help="Path to input image")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug images")
    parser.add_argument("--mode", default="oll", choices=["oll"], help="Recognition mode")
    parser.add_argument("--crop", action="store_true", help="Save 3x3 individual cell crops")
    parser.add_argument("--grid-image", action="store_true", help="Save 300x300 annotated warped grid image")
    parser.add_argument("--validate", action="store_true", help="Run validation against dataset/labels.json")
    args = parser.parse_args()

    if args.validate:
        return validate_dataset(debug=args.debug, save_crops=args.crop, save_grid=args.grid_image)

    if not args.image_path:
        print(json.dumps({"error": "image_path is required unless --validate is set"}))
        return 1

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        return 1

    oll_algos = parse_oll_algorithms(Path("references/oll.md"))
    pattern_lookup = build_oll_pattern_lookup()

    try:
        colors, oriented, base_confidence, _, _, artifacts = detect_and_classify(
            image_path,
            args.debug,
            save_crops=args.crop,
            save_grid=args.grid_image,
        )
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
    }
    result.update(artifacts)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
