import os
import cv2
import numpy as np
import pandas as pd

RAW_TRAIN = "data/raw/train_images"
RAW_VAL = "data/raw/val_images"

TRAIN_CSV = "data/raw/train_labels.csv"
VAL_CSV = "data/raw/val_labels.csv"

OUT_DIR = "data/processed/images"
LABELS_OUT = "data/processed/labels.txt"

os.makedirs(OUT_DIR, exist_ok=True)

def process_and_save(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    new_h = 32
    new_w = int((w / h) * new_h)

    resized = cv2.resize(gray, (new_w, new_h))

    padded = 255 * np.ones((32, 128), dtype=np.uint8)
    padded[:, :min(new_w, 128)] = resized[:, :128]

    cv2.imwrite(save_path, padded)
    return True


def process_dataset(csv_path, image_dir, prefix):
    df = pd.read_csv(csv_path)

    processed = []

    for _, row in df.iterrows():
        img_name = row["IMAGE"]
        label = str(row["MEDICINE_NAME"]).lower()

        src_path = os.path.join(image_dir, img_name)

        # 🔥 make filename UNIQUE
        new_name = f"{prefix}_{img_name}"
        dst_path = os.path.join(OUT_DIR, new_name)

        success = process_and_save(src_path, dst_path)

        if success:
            processed.append((new_name, label))
        else:
            print("❌ Failed:", src_path)

    return processed


print("🚀 Processing TRAIN dataset...")
train_data = process_dataset(TRAIN_CSV, RAW_TRAIN, "train")

print("🚀 Processing VAL dataset...")
val_data = process_dataset(VAL_CSV, RAW_VAL, "val")

all_data = train_data + val_data

print("Total processed images:", len(all_data))

# 🔥 write labels
with open(LABELS_OUT, "w", encoding="utf-8") as f:
    for img, label in all_data:
        f.write(f"{img} {label}\n")

print("✅ labels.txt created")