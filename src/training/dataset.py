import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

# 🔹 Character set
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"

char2idx = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
idx2char = {i: c for c, i in char2idx.items()}


def encode_label(text):
    return [char2idx[c] for c in text if c in char2idx]


# 🔥 ADVANCED CURSIVE AUGMENTATION
def elastic_transform(image, alpha=15, sigma=4):
    shape = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return distorted

def augment(img):
    # 1. Morphological thickness variation (pen stroke adjustment)
    if random.random() < 0.4:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            # Erode shrinks white background, making dark ink THICKER
            img = cv2.erode(img, kernel, iterations=1)
        else:
            # Dilate expands white background, making dark ink THINNER
            img = cv2.dilate(img, kernel, iterations=1)

    # 2. Elastic distortion (curves and warps the cursive script randomly)
    if random.random() < 0.5:
        img = elastic_transform(img, alpha=random.randint(10, 20), sigma=4)

    # 3. Dynamic Brightness
    if random.random() < 0.5:
        img = img * (0.7 + 0.6 * random.random())

    # 4. Ambient static noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0, 1).astype(np.float32)
    return img


class OCRDataset(Dataset):
    def __init__(self, labels_path, images_dir):
        self.images_dir = images_dir
        self.samples = []

        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                fname = parts[0]
                label = "".join(parts[1:])

                self.samples.append((fname, label))

        if len(self.samples) == 0:
            raise RuntimeError("❌ No valid samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]

        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"❌ Image not found: {img_path}")

        # Mathematically resize and pad ALL incoming images to strictly [32x128]
        h, w = img.shape
        new_w = int(w * (32 / h))
        if new_w > 128:
            new_w = 128
        
        img = cv2.resize(img, (new_w, 32))
        img = cv2.copyMakeBorder(img, 0, 0, 0, 128 - new_w, cv2.BORDER_CONSTANT, value=255)

        img = img.astype(np.float32) / 255.0

        img = augment(img)

        img = np.expand_dims(img, axis=0).astype(np.float32)  # 🔥 FORCE

        label_encoded = encode_label(label)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label_encoded, dtype=torch.long)


def collate_fn(batch):
    images, labels = zip(*batch)

    images = torch.stack(images)

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels_concat = torch.cat(labels)

    return images, labels_concat, label_lengths