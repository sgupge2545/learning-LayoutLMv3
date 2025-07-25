#!/usr/bin/env python3
"""
LayoutLMv3 用 学習データ自動生成スクリプト（日本語）
--------------------------------------------------
• 複数画像を読み込み → 書類領域を自動トリミング（机など背景をカット）
• Tesseract で OCR（words / bboxes 取得）
• 画像ファイル名 (xxxx.jpg) から車両番号文字列を取得し、自動でラベル付け
    - ラベルは "B-VEHICLE_NUM"（1トークン想定）
• 最終的に train.json を出力

前提:
  brew install tesseract tesseract-lang  # 日本語辞書
  pip install opencv-python pillow pytesseract transformers datasets

使い方:
  python layoutlmv3_dataset_builder.py \
      --input_dir ./images \
      --output_dir ./cropped \
      --json_path ./train.json

注意:
  - OCR が車両番号を正しく分割しない場合、正規表現で再結合してヒットさせています。
  - 見つけられなかった場合は、ページ全体 bbox にラベルを貼る fallback を実装しています。
  - 必要に応じて正規表現や正規化関数を調整してください。
"""

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

import tkinter as tk
from tkinter import filedialog

# -----------------------------
# Utils
# -----------------------------


def four_point_transform(
    image: np.ndarray, pts: np.ndarray, out_w: int = 1000, out_h: int = 1400
) -> np.ndarray:
    """透視変換で書類を長方形に切り出す。pts は 4x2 の順序付き座標。
    out_w/out_h は出力画像サイズ（レイアウト学習用に固定）。"""
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


def detect_document(img: np.ndarray) -> np.ndarray:
    """最大輪郭から四角形近似して 4 点を返す。見つからなければ None。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 75, 200)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return None

    # 順序付け（左上, 右上, 右下, 左下）
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def normalize_bbox(box, width, height):
    x, y, w, h = box
    return [
        int(x * 1000 / width),
        int(y * 1000 / height),
        int((x + w) * 1000 / width),
        int((y + h) * 1000 / height),
    ]


def normalize_text(s: str) -> str:
    """比較用に正規化（全角→半角、空白・記号除去など、必要に応じ調整）"""
    import unicodedata

    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    return s


def find_value_span(words, value_norm):
    """OCR で得た words リストから、value_norm（正規化済文字列）に
    一致する連続トークン範囲を返す。見つからなければ (-1, -1)。"""
    norm_words = [normalize_text(w) for w in words]
    for i in range(len(norm_words)):
        acc = ""
        for j in range(i, len(norm_words)):
            acc += norm_words[j]
            if acc == value_norm:
                return i, j
            if len(acc) > len(value_norm):
                break
    return -1, -1


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / max(areaA + areaB - inter, 1)


# -----------------------------
# Main
# -----------------------------


def select_image_from_origin():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir="origin",
        title="画像を選択",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")],
    )
    return file_path


def main():
    input_dir = "origin"
    output_json = "dataset.json"
    annotations = []

    img_paths = sorted(
        [
            p
            for p in Path(input_dir).glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        ]
    )

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        clone = img.copy()
        bbox = []
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(clone, (x, y), 4, (0, 255, 0), -1)
                cv2.imshow("select bbox", clone)
                if len(points) == 2:
                    cv2.rectangle(clone, points[0], points[1], (0, 0, 255), 2)
                    cv2.imshow("select bbox", clone)

        cv2.imshow("select bbox", clone)
        cv2.setMouseCallback("select bbox", click_event)
        print(f"{img_path.name}：左上→右下の順に2回クリックしてください")
        while True:
            key = cv2.waitKey(1)
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)
                bbox = [x_min, y_min, x_max, y_max]
                break
            if key == 27:  # ESCでスキップ
                print("スキップ")
                bbox = [0, 0, img.shape[1], img.shape[0]]
                break
        cv2.destroyAllWindows()

        stem = img_path.stem
        h, w = img.shape[:2]
        bbox_norm = [
            int(x_min * 1000 / w),
            int(y_min * 1000 / h),
            int(x_max * 1000 / w),
            int(y_max * 1000 / h),
        ]
        trimed_dir = "trimed"
        os.makedirs(trimed_dir, exist_ok=True)
        img_with_rect = img.copy()
        cv2.rectangle(img_with_rect, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        save_path = os.path.join(trimed_dir, img_path.name)
        cv2.imwrite(save_path, img_with_rect)

        veh_word = stem
        veh_label = "B-VEHICLE_NUM"
        veh_box_norm = bbox_norm

        pil_img = Image.open(img_path).convert("RGB")
        ocr = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT, lang="jpn"
        )
        W, H = pil_img.size

        words = [veh_word]
        bboxes = [veh_box_norm]
        labels = [veh_label]

        for i in range(len(ocr["text"])):
            txt = ocr["text"][i].strip()
            if txt == "" or int(ocr["conf"][i]) <= 0:
                continue
            x, y, w_box, h_box = (
                ocr["left"][i],
                ocr["top"][i],
                ocr["width"][i],
                ocr["height"][i],
            )
            norm_box = [
                int(x * 1000 / W),
                int(y * 1000 / H),
                int((x + w_box) * 1000 / W),
                int((y + h_box) * 1000 / H),
            ]
            if iou(norm_box, veh_box_norm) < 0.3:
                words.append(txt)
                bboxes.append(norm_box)
                labels.append("O")

        ann = {
            "id": stem,
            "image_path": str(img_path),  # こうする（元画像の絶対/相対パス）
            "words": words,
            "bboxes": bboxes,
            "labels": labels,
        }
        annotations.append(ann)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f"✅ 完了: {len(annotations)} 件を書き出し → {output_json}")


if __name__ == "__main__":
    main()
