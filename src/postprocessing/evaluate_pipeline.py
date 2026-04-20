import torch
import cv2
import numpy as np
import pandas as pd
import json

from src.training.model import CRNN
from src.postprocessing.final_pipeline import final_match

# =========================
# CONFIG
# =========================
MODEL_PATH = "model_best.pth"
LABEL_PATH = "data/processed/labels_balanced.txt"
IMAGE_DIR = "data/processed/images_balanced"

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

# =========================
# LOAD MODEL
# =========================
model = CRNN(len(CHARS) + 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# =========================
# PREPROCESS
# =========================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

# =========================
# TOP-K DECODE
# =========================
def decode_topk(output, k=5):
    probs = torch.softmax(output, dim=2)
    topk = torch.topk(probs, k, dim=2).indices

    results = []

    for i in range(k):
        seq = topk[0, :, i]

        text = ""
        prev = -1

        for t in seq:
            t = t.item()
            if t != 0 and t != prev:
                text += idx2char.get(t, "")
            prev = t

        results.append(text)

    return results

# =========================
# EVALUATION (LABEL LEVEL)
# =========================
correct = 0
total = 0

with open(LABEL_PATH, "r") as f:
    lines = f.readlines()

with open("data/processed/generic_to_brand.json", "r") as f:
    generic_to_brand = json.load(f)

brand_to_generic = {}
for generic, brands in generic_to_brand.items():
    brand_to_generic[generic] = generic
    for brand in brands:
        brand_to_generic[brand] = generic

print("\n[RESULT] LABEL-LEVEL PIPELINE EVALUATION:\n")

for line in lines[:200]:  # test on 200 samples
    fname, gt_label = line.strip().split(maxsplit=1)
    gt_label = gt_label.lower()

    path = f"{IMAGE_DIR}/{fname}"

    img = preprocess(path)

    with torch.no_grad():
        out = model(img)

    preds = decode_topk(out, k=5)

    # 🔥 get predicted label using FINAL MATCH intelligent logic
    match_result = final_match(preds)
    predicted_label = match_result["generic"] if match_result else ""

    gt_generic = brand_to_generic.get(gt_label, gt_label)
    pred_generic = brand_to_generic.get(predicted_label, predicted_label)
    
    # Allow a match if they resolve to the exact same base generic medication computationally
    if gt_generic == pred_generic and pred_generic != "":
        correct += 1
    else:
        print(f"FAILED: gt='{gt_label}' (gen '{gt_generic}'), pred='{predicted_label}' (gen '{pred_generic}')")

    total += 1

accuracy = (correct / total) * 100

print(f"Total Samples: {total}")
print(f"Final Label Accuracy: {accuracy:.2f}%")