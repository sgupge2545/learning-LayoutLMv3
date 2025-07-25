import json
import pytesseract
from PIL import Image, ImageOps
import os


def create_test_json(image_dir="test_images", output_file="test_input.json"):
    """
    指定ディレクトリの画像をOCRして、test_input.jsonを作成
    """
    test_data = []

    # 画像ファイルを取得
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # OCR実行
        ocr = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, lang="jpn"
        )
        W, H = image.size

        words = []
        bboxes = []

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
            words.append(txt)
            bboxes.append(norm_box)

        # サンプルデータに追加
        sample = {
            "id": os.path.splitext(img_file)[0],  # 拡張子除く
            "image_path": img_path,
            "words": words,
            "bboxes": bboxes,
        }
        test_data.append(sample)

    # JSON保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"{len(test_data)}件の画像を処理して{output_file}を作成しました")


if __name__ == "__main__":
    create_test_json()
