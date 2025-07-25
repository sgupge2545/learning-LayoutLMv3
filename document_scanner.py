import cv2
import numpy as np
from PIL import Image


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_document_edges(image: np.ndarray, max_height: int = 1000) -> np.ndarray:
    # 速度確保のため長辺を揃えてダウンサンプリング
    ratio = 1.0
    if image.shape[0] > max_height:
        ratio = image.shape[0] / max_height
        image_small = cv2.resize(image, (int(image.shape[1] / ratio), max_height))
    else:
        image_small = image.copy()
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx.reshape(4, 2)
            break
        # 4点でなければ最小外接四辺形で近似
        elif len(approx) > 4:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            doc_cnt = np.int0(box)
            break
    if doc_cnt is None:
        return None, ratio
    # 小さい画像で検出した座標を元解像度に戻す
    doc_cnt = doc_cnt * ratio
    rect = order_points(doc_cnt)
    return rect, ratio


def perspective_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = points
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxWidth = int(max(wA, wB))
    maxHeight = int(max(hA, hB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document_edges_allow_out(
    image: np.ndarray, max_height: int = 1000, pad: int = 200
):
    """
    画像外にはみ出た四角形でも検出できるよう、先に余白を付けてから検出する。
    """
    # 1. 先にパディング
    img_pad = cv2.copyMakeBorder(
        image, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # 2. 既存の関数を使って検出
    rect_pad, ratio = detect_document_edges(img_pad, max_height=max_height)
    if rect_pad is None:
        return None, image  # 失敗時は元画像を返す

    # 3. 余白分を戻す
    rect = rect_pad - np.array([pad, pad])

    # 4. もし戻した結果がネガティブなら、再度パディングして安全化
    img_safe, rect_safe = pad_image_to_include_points(image, rect)

    return rect_safe, img_safe


def scan_document(pil_image: Image.Image, max_height: int = 1000) -> Image.Image:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    rect, image_used = detect_document_edges_allow_out(
        image, max_height=max_height, pad=2000
    )
    if rect is None:
        print("書類らしき四角形が見つかりませんでした")
        return pil_image

    warped = perspective_transform(image_used, rect.astype(np.float32))
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


def enhance_document_quality(
    image: Image.Image,
    blur: str = None,  # 'gaussian', 'median', None
    binarize: bool = True,  # 2値化するか
    clahe: bool = False,  # コントラスト強調するか
    adaptive_block_size: int = 25,  # adaptiveThresholdのblockSize
    adaptive_C: int = 40,  # adaptiveThresholdのC
    global_thresh: int = None,  # 大域的2値化の閾値（NoneならadaptiveThreshold）
) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if blur == "gaussian":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif blur == "median":
        gray = cv2.medianBlur(gray, 3)

    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_obj.apply(gray)

    if binarize:
        if global_thresh is not None:
            _, gray = cv2.threshold(gray, global_thresh, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                adaptive_block_size,
                adaptive_C,
            )

    return Image.fromarray(gray)


def detect_document_by_threshold(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 大域的なしきい値
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 反転しておく（書類が白なら黒背景に）
    thresh = 255 - thresh
    # 輪郭抽出
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # 最大領域を選択
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def pad_image_to_include_points(image: np.ndarray, points: np.ndarray):
    h, w = image.shape[:2]
    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    max_x = np.max(points[:, 0])
    max_y = np.max(points[:, 1])

    pad_left = max(0, -int(np.floor(min_x)))
    pad_top = max(0, -int(np.floor(min_y)))
    pad_right = max(0, int(np.ceil(max_x)) - w + 1)
    pad_bottom = max(0, int(np.ceil(max_y)) - h + 1)

    image_padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],  # 白でパディング（書類背景色に合わせて調整）
    )
    points_padded = points + np.array([pad_left, pad_top])
    return image_padded, points_padded
