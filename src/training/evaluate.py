import torch
import cv2
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from src.training.model import CRNN

# =========================
# 🔹 CONFIG
# =========================
MODEL_PATH = "model.pth"
LABEL_PATH = "data/processed/labels_balanced.txt"
IMAGE_DIR = "data/processed/images_balanced"

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

# =========================
# 🔹 LOAD MODEL
# =========================
model = CRNN(len(CHARS) + 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# =========================
# 🔹 PREPROCESS
# =========================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

# =========================
# 🔹 DECODE
# =========================
def decode(output):
    pred = output.argmax(2)[0]

    text = ""
    prev = -1

    for t in pred:
        t = t.item()
        if t != 0 and t != prev:
            text += idx2char.get(t, "")
        prev = t

    return text

# =========================
# 🔹 EVALUATION
# =========================
correct = 0
total = 0
char_scores = []

with open(LABEL_PATH, "r") as f:
    lines = f.readlines()

for line in lines[:500]:   # 🔥 test on 500 samples first
    fname, gt = line.strip().split(maxsplit=1)

    path = f"{IMAGE_DIR}/{fname}"

    img = preprocess(path)

    with torch.no_grad():
        out = model(img)

    pred = decode(out)

    # 🔹 word accuracy
    if pred == gt:
        correct += 1

    # 🔹 character similarity
    score = fuzz.ratio(pred, gt)
    char_scores.append(score)

    total += 1

# =========================
# 🔹 RESULTS
# =========================
word_acc = (correct / total) * 100
char_acc = sum(char_scores) / len(char_scores)

print("\n[RESULT] OCR EVALUATION RESULT:\n")
print(f"Total Samples: {total}")
print(f"Word Accuracy: {word_acc:.2f}%")
print(f"Character Accuracy: {char_acc:.2f}%")