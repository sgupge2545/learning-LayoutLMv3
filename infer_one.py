import json
import torch
import os
from PIL import Image, ImageOps
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch.nn.functional as F

LABEL_LIST = ["O", "B-VEHICLE_NUM"]

# モデル・プロセッサのロード
processor = LayoutLMv3Processor.from_pretrained("./model_out")
processor.image_processor.apply_ocr = False
model = LayoutLMv3ForTokenClassification.from_pretrained("./model_out")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# val.json全件推論
with open("val.json", encoding="utf-8") as f:
    val_data = json.load(f)

for sample in val_data:
    image = Image.open(sample["image_path"])
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    enc = processor(
        images=image,
        text=sample["words"],
        boxes=sample["bboxes"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    word_ids = enc.word_ids(batch_index=0)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.cpu()
    pred_ids = logits.argmax(-1)[0].tolist()
    # 単語単位に多数決
    word_pred = {}
    for tid, wid in zip(pred_ids, word_ids):
        if wid is None:
            continue
        lbl = LABEL_LIST[tid]
        if wid not in word_pred:
            word_pred[wid] = []
        word_pred[wid].append(lbl)
    pred_labels = []
    for wid in range(len(sample["words"])):
        labs = word_pred.get(wid, ["O"])
        label = max(set(labs), key=labs.count)
        pred_labels.append(label)
    # ここでpred_labelsを使って何かしたい場合はprintなどで出力
    print(f"{sample['id']}: {pred_labels}")

    # 2. **どの単語が「B-VEHICLE_NUM」と推定されたかを見たい場合**
    for wid, (word, label) in enumerate(zip(sample["words"], pred_labels)):
        print(f"{word}: {label}")

    # 3. **確信度（スコア、確率）も見たい場合**
    probs = F.softmax(logits, dim=-1)[0]  # shape: seq_len x num_labels
    for wid, word in enumerate(sample["words"]):
        # word_idsでサブワード→単語対応
        indices = [i for i, w in enumerate(word_ids) if w == wid]
        if not indices:
            continue
        # 各サブワードの最大確信度
        confs = [probs[i, pred_ids[i]].item() for i in indices]
        label = pred_labels[wid]
        print(f"{word}: {label} (conf={max(confs):.3f})")
