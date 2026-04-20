import torch
import cv2
import numpy as np

from src.training.model import CRNN
from src.pipeline.segment_lines import segment_lines

MODEL_PATH = "model.pth"

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

model = CRNN(len(CHARS) + 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def preprocess(img):
    img = cv2.resize(img, (128, 32))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

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

def debug_pipeline(image_path):

    lines = segment_lines(image_path)

    print("\n🔍 DEBUG: OCR OUTPUT\n")

    for i, line_path in enumerate(lines):

        print(f"\n--- Line {i} ---")

        line_img = cv2.imread(line_path, cv2.IMREAD_GRAYSCALE)

        if line_img is None:
            continue

        _, thresh = cv2.threshold(line_img, 150, 255, cv2.THRESH_BINARY_INV)

        vertical_sum = np.sum(thresh, axis=0)
        splits = np.where(vertical_sum < 5)[0]

        tokens = []
        prev = 0

        for s in splits:
            if s - prev > 20:
                word = line_img[:, prev:s]
                tokens.append(word)
            prev = s

        if line_img.shape[1] - prev > 20:
            tokens.append(line_img[:, prev:])

        for j, token_img in enumerate(tokens):

            img_tensor = preprocess(token_img)

            with torch.no_grad():
                out = model(img_tensor)

            preds = decode_topk(out, k=5)

            print(f"Token {j}: {preds}")
            print("-" * 40)

if __name__ == "__main__":
    image_path = "test_prescription.jpg"
    debug_pipeline(image_path)