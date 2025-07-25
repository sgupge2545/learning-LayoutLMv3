import cv2
import json
import os
from tkinter import filedialog, Tk

output = []

print("=== LayoutLMv3 アノテーションツール ===")

# Tkinterでファイル選択ダイアログを開く
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="画像ファイルを選択", filetypes=[("Image files", "*.jpg *.png")]
)

if not image_path:
    print("キャンセルされました")
    exit()

filename = os.path.basename(image_path)
img = cv2.imread(image_path)
clone = img.copy()
bboxes = []

# グローバル変数
selecting = False
x_start, y_start = -1, -1


def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, selecting

    if event == cv2.EVENT_LBUTTONDOWN:
        if not selecting:
            # 1点目
            x_start, y_start = x, y
            selecting = True
        else:
            # 2点目
            x_end, y_end = x, y
            selecting = False

            x1, y1 = min(x_start, x_end), min(y_start, y_end)
            x2, y2 = max(x_start, x_end), max(y_start, y_end)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bboxes.append(
                [
                    x1 * 1000 // img.shape[1],
                    y1 * 1000 // img.shape[0],
                    x2 * 1000 // img.shape[1],
                    y2 * 1000 // img.shape[0],
                ]
            )


print("画像上で車両番号をドラッグで囲ってください（ESCで終了）")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_rectangle)

while True:
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESCキー
        break

cv2.destroyAllWindows()

if not bboxes:
    print("矩形が1つも登録されていません。終了します。")
    exit()

# CLIで内容・ラベルを入力
words = []
labels = []
for i, bbox in enumerate(bboxes):
    word = input(f"{i + 1}つ目のbboxの内容を入力してください: ").strip()
    label = input(f"{i + 1}つ目のbboxのラベルを入力してください: ").strip()
    words.append(word)
    labels.append(label)

data = {
    "id": os.path.splitext(filename)[0],
    "image_path": image_path,
    "words": words,
    "bboxes": bboxes,
    "labels": labels,
}

output_path = "train.json"
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
else:
    existing = []

existing.append(data)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(existing, f, ensure_ascii=False, indent=2)

print(f"✅ アノテーション完了！ → {output_path} に保存しました")
