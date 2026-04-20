import torch
import cv2
import numpy as np
from src.training.model import CRNN

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

num_classes = len(CHARS) + 1

model = CRNN(num_classes)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)


def decode_topk(output, k=5):
    probs = torch.softmax(output, dim=2)

    topk = torch.topk(probs, k, dim=2).indices  # (B, W, K)

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


# 🔹 TEST
import os

folder = "data/processed/images_balanced"
files = os.listdir(folder)[:5]

print("\n🔍 TOP-K OCR OUTPUT:\n")

for f in files:
    path = os.path.join(folder, f)

    img = preprocess(path)

    with torch.no_grad():
        out = model(img)
        preds = decode_topk(out, k=5)

    print(f"\n{f}:")
    for p in preds:
        print("  →", p)