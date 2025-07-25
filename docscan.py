#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLE-Hough: Outer Line Extraction with Hough

- ρ–θ HoughLines で直線取得
- 角度で水平/垂直をフィルタ
- 端から近い順に複数候補を用意し、組合せで四角形生成
- 幾何スコア(面積・直角) + 白さスコア(Lab)で最良を選択
- 透視変換して保存
- --debug で lines/edges/warped を保存

Requirements: opencv-python, numpy, pillow
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image


# -------------------- 基本ユーティリティ --------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rho_theta_to_abc(rho: float, theta: float) -> np.ndarray:
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    n = np.hypot(a, b)
    return np.array([a / n, b / n, c / n], dtype=np.float64)


def intersect(L1: np.ndarray, L2: np.ndarray):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    D = a1 * b2 - a2 * b1
    if abs(D) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / D
    y = (c1 * a2 - c2 * a1) / D
    return np.array([x, y], dtype=np.float32)


def perspective_warp(image_bgr: np.ndarray, rect: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = rect.astype(np.float32)
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))
    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    return cv2.warpPerspective(image_bgr, M, (maxW, maxH))


# -------------------- 線選択ヘルパ --------------------
def line_angle_deg(eq):
    a, b, _ = eq
    ang = np.degrees(np.arctan2(a, -b))  # 0°:水平, 90°:垂直
    return (ang + 180) % 180


def filter_by_angle(eqs, orient, tol_deg=5):
    out = []
    for eq in eqs:
        ang = line_angle_deg(eq)
        if orient == "horizontal" and abs(ang - 0) <= tol_deg:
            out.append(eq)
        elif orient == "vertical" and abs(ang - 90) <= tol_deg:
            out.append(eq)
    return np.array(out)


def side_score(eq, side: str, W: int, H: int, samples: int = 9) -> float:
    a, b, c = eq
    if side in ("left", "right"):
        ys = np.linspace(0, H - 1, samples)
        xs = (-b * ys - c) / (a + 1e-9)
        return xs.min() if side == "left" else xs.max()
    else:
        xs = np.linspace(0, W - 1, samples)
        ys = (-a * xs - c) / (b + 1e-9)
        return ys.min() if side == "top" else ys.max()


def pick_outer_lines_try(eqs, orient, W, H, max_try=3):
    if orient == "vertical":
        left_sorted = np.argsort(
            [side_score(eq, "left", W, H) for eq in eqs]
        )  # small -> big
        right_sorted = np.argsort([side_score(eq, "right", W, H) for eq in eqs])[
            ::-1
        ]  # big   -> small
    else:
        top_sorted = np.argsort(
            [side_score(eq, "top", W, H) for eq in eqs]
        )  # small -> big
        bottom_sorted = np.argsort([side_score(eq, "bottom", W, H) for eq in eqs])[::-1]

    pairs = []
    for i in range(min(max_try, len(eqs))):
        for j in range(min(max_try, len(eqs))):
            if orient == "vertical":
                li = left_sorted[i]
                rj = right_sorted[j]
                if li == rj:
                    continue
                pairs.append((eqs[li], eqs[rj]))
            else:
                ti = top_sorted[i]
                bj = bottom_sorted[j]
                if ti == bj:
                    continue
                pairs.append((eqs[ti], eqs[bj]))
    return pairs


# -------------------- スコアリング --------------------
def quad_score(rect_small: np.ndarray) -> float:
    """幾何的妥当性: 面積大 + 直角性が高いほど良い"""
    area = cv2.contourArea(rect_small.astype(np.float32))
    if area <= 0:
        return -1e9
    ang_pen = 0.0
    for i in range(4):
        v1 = rect_small[(i + 1) % 4] - rect_small[i]
        v2 = rect_small[(i - 1) % 4] - rect_small[i]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        ang_pen += abs(ang - 90)
    return area - 10 * ang_pen  # 係数は状況で調整


def whiteness_score(
    image_bgr: np.ndarray, rect_full: np.ndarray, margin_ratio=0.03
) -> float:
    """紙の白さ/均一さ + 外周の暗さで評価"""
    warped = perspective_warp(image_bgr, rect_full.astype(np.float32))
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)

    h, w = L.shape
    m = int(min(h, w) * margin_ratio)
    if m < 1:
        return 0.0

    inner = L[m : h - m, m : w - m]
    borders = np.concatenate(
        [
            L[:m, :].ravel(),
            L[h - m :, :].ravel(),
            L[:, :m].ravel(),
            L[:, w - m :].ravel(),
        ]
    )

    inner_mean = inner.mean()
    inner_std = inner.std()
    border_mean = borders.mean()

    # 白くて均一(=std小)で、境界より明るいほど高スコア
    score = (inner_mean - 0.5 * inner_std) - 0.2 * border_mean
    return float(score)


# -------------------- デバッグ描画 --------------------
def draw_debug_lines(img_small, rect_small, v_left, v_right, h_top, h_bot):
    out = img_small.copy()
    H, W = img_small.shape[:2]

    def draw_line(eq, color):
        a, b, c = eq
        if abs(b) > abs(a):
            x0, x1 = 0, W
            y0 = int((-a * x0 - c) / (b + 1e-9))
            y1 = int((-a * x1 - c) / (b + 1e-9))
            cv2.line(out, (x0, y0), (x1, y1), color, 2)
        else:
            y0, y1 = 0, H
            x0 = int((-b * y0 - c) / (a + 1e-9))
            x1 = int((-b * y1 - c) / (a + 1e-9))
            cv2.line(out, (x0, y0), (x1, y1), color, 2)

    draw_line(v_left, (0, 0, 255))
    draw_line(v_right, (0, 0, 255))
    draw_line(h_top, (0, 255, 255))
    draw_line(h_bot, (0, 255, 255))

    for p in rect_small:
        cv2.circle(out, tuple(np.int32(p)), 6, (255, 0, 0), -1)
    return out


# -------------------- 検出本体 --------------------
def detect_outer_box(
    image_bgr: np.ndarray,
    max_height: int = 1400,
    canny1: int = 50,
    canny2: int = 150,
    hough_thresh: int = 220,
    angle_eps_deg: float = 5.0,
    angle_tol_deg: float = 5.0,
    try_k: int = 3,
    debug: bool = False,
    debug_dir: str = None,
    stem: str = "img",
):
    ratio = 1.0
    if image_bgr.shape[0] > max_height:
        ratio = image_bgr.shape[0] / max_height
        img = cv2.resize(image_bgr, (int(image_bgr.shape[1] / ratio), max_height))
    else:
        img = image_bgr.copy()

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny1, canny2)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)
    if lines is None or len(lines) < 4:
        return None, {"reason": "not enough lines"}

    # θで分類し、さらに角度でフィルタ
    verts_eq, hors_eq = [], []
    for rho, theta in lines[:, 0]:
        deg = np.degrees(theta)
        if deg < angle_eps_deg or deg > 180 - angle_eps_deg:
            verts_eq.append(rho_theta_to_abc(rho, theta))
        elif abs(deg - 90) < angle_eps_deg:
            hors_eq.append(rho_theta_to_abc(rho, theta))

    verts_eq = filter_by_angle(verts_eq, "vertical", tol_deg=angle_tol_deg)
    hors_eq = filter_by_angle(hors_eq, "horizontal", tol_deg=angle_tol_deg)
    if len(verts_eq) < 2 or len(hors_eq) < 2:
        return None, {"reason": "filtered too many"}

    best = None
    best_score = -1e9

    for vL, vR in pick_outer_lines_try(verts_eq, "vertical", W, H, max_try=try_k):
        for hT, hB in pick_outer_lines_try(hors_eq, "horizontal", W, H, max_try=try_k):
            pts = [
                intersect(hT, vL),
                intersect(hT, vR),
                intersect(hB, vR),
                intersect(hB, vL),
            ]
            if any(p is None for p in pts):
                continue
            rect_small = order_points(np.array(pts))
            rect_full = rect_small * ratio

            geo_s = quad_score(rect_small)
            white_s = whiteness_score(image_bgr, rect_full)

            total = geo_s + white_s * 10  # 重み適当に。調整可
            if total > best_score:
                best_score = total
                best = (rect_small, vL, vR, hT, hB)

    if best is None:
        return None, {"reason": "no valid quad"}

    rect_small, v_left_eq, v_right_eq, h_top_eq, h_bot_eq = best
    rect = rect_small * ratio

    if debug:
        os.makedirs(debug_dir or "debug_out", exist_ok=True)
        dbg_img = draw_debug_lines(
            img, rect_small, v_left_eq, v_right_eq, h_top_eq, h_bot_eq
        )
        cv2.imwrite(
            os.path.join(debug_dir or "debug_out", f"{stem}_lines.jpg"), dbg_img
        )
        cv2.imwrite(os.path.join(debug_dir or "debug_out", f"{stem}_edges.png"), edges)

    return rect, {}


# -------------------- High-level API --------------------
def scan_document_pil(
    pil_img: Image.Image, debug: bool = False, debug_dir: str = None, stem: str = "img"
) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    rect, info = detect_outer_box(bgr, debug=debug, debug_dir=debug_dir, stem=stem)
    if rect is None:
        print("書類らしき四角形が見つかりませんでした:", info.get("reason"))
        return pil_img
    warped = perspective_warp(bgr, rect)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


# -------------------- Batch --------------------
def is_img(name: str) -> bool:
    return name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))


def batch_process(in_dir: str, out_dir: str, debug: bool):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(in_dir):
        if not is_img(fn):
            continue
        ip = os.path.join(in_dir, fn)
        op = os.path.join(out_dir, fn)
        try:
            stem = os.path.splitext(fn)[0]
            img = Image.open(ip)
            scanned = scan_document_pil(img, debug=debug, debug_dir=out_dir, stem=stem)
            scanned.save(op)
            print(f"[OK] {fn}")
        except Exception as e:
            print(f"[NG] {fn}: {e}")


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="file or folder")
    ap.add_argument("-o", "--output", required=True, help="file or folder")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        batch_process(args.input, args.output, args.debug)
    else:
        img = Image.open(args.input)
        stem = os.path.splitext(os.path.basename(args.input))[0]
        out_img = scan_document_pil(
            img,
            debug=args.debug,
            debug_dir=os.path.dirname(args.output) or ".",
            stem=stem,
        )
        out_img.save(args.output)
        print(f"[OK] {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
