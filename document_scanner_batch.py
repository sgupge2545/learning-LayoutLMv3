import os
from PIL import Image
from document_scanner import scan_document

origin_dir = "origin"
scaned_dir = "scanned"

if not os.path.exists(scaned_dir):
    os.makedirs(scaned_dir)

for filename in os.listdir(origin_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        input_path = os.path.join(origin_dir, filename)
        output_path = os.path.join(scaned_dir, filename)
        try:
            img = Image.open(input_path)
            scanned_img = scan_document(img)
            scanned_img.save(output_path)
            print(f"{filename} をスキャンして保存しました")
        except Exception as e:
            print(f"{filename} の処理中にエラー: {e}")
