# train_layoutlmv3.py
import json
from PIL import Image, ImageDraw
from datasets import Dataset
import torch
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)

LABEL_LIST = ["O", "B-VEHICLE_NUM"]
id2label = {i: l for i, l in enumerate(LABEL_LIST)}
label2id = {l: i for i, l in id2label.items()}

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
processor.image_processor.apply_ocr = False  # ★これを追加
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id,
)


def load_ds(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    return Dataset.from_list(data)


def preprocess(batch):
    image = Image.open(batch["image_path"]).convert("RGB")
    word_labels = [label2id[l] for l in batch["labels"]]

    enc = processor(
        images=image,
        text=batch["words"],  # ★ words -> text に
        boxes=batch["bboxes"],
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in enc.items()}


train_ds = load_ds("train.json").map(preprocess)
val_ds = load_ds("val.json").map(preprocess)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

args = TrainingArguments(
    output_dir="./model_out",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor.tokenizer,
)

trainer.train()
trainer.save_model("./model_out")
processor.save_pretrained("./model_out")


def save_prediction_visualization(sample, pred_labels, save_path):
    # sample: 1件分のdict（image_path, words, bboxes, ...）
    # pred_labels: 単語ごとの予測ラベル（例: ["O", "B-VEHICLE_NUM", ...]）
    image = Image.open(sample["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)

    w, h = image.size
    for word, bbox, label in zip(sample["words"], sample["bboxes"], pred_labels):
        if label == "B-VEHICLE_NUM":
            # bboxが0-1000正規化なら元画像サイズに変換
            x1 = int(bbox[0] * w / 1000)
            y1 = int(bbox[1] * h / 1000)
            x2 = int(bbox[2] * w / 1000)
            y2 = int(bbox[3] * h / 1000)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), word, fill="red")

    image.save(save_path)
    print(f"保存しました: {save_path}")
