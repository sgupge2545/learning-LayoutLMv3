import json
from PIL import Image, ImageOps
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch.nn.functional as F

LABEL_LIST = ["O", "B-VEHICLE_NUM"]

processor = LayoutLMv3Processor.from_pretrained("./model_out")
processor.image_processor.apply_ocr = False
model = LayoutLMv3ForTokenClassification.from_pretrained("./model_out")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

with open("test_input.json", encoding="utf-8") as f:
    test_data = json.load(f)

all_results = []

for sample in test_data:
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

    probs = F.softmax(logits, dim=-1)[0]
    results = []
    for wid, word in enumerate(sample["words"]):
        indices = [i for i, w in enumerate(word_ids) if w == wid]
        if not indices:
            continue
        confs = [probs[i, pred_ids[i]].item() for i in indices]
        label = pred_labels[wid]
        results.append({"word": word, "label": label, "confidence": max(confs)})
    all_results.append({"id": sample["id"], "results": results})

with open("test_value.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
