# split_dataset.py
import json
import random

random.seed(42)

src = "dataset.json"  # あなたの JSON
data = json.load(open(src, "r", encoding="utf-8"))

# sanity check
for i, d in enumerate(data):
    assert len(d["words"]) == len(d["bboxes"]) == len(d["labels"]), (
        f"len mismatch at {i}"
    )
    assert all(0 <= v <= 1000 for box in d["bboxes"] for v in box), (
        f"bbox range error at {i}"
    )

random.shuffle(data)
n = int(len(data) * 0.8)
json.dump(
    data[:n], open("train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2
)
json.dump(
    data[n:], open("val.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2
)
print("train:", n, "val:", len(data) - n)
