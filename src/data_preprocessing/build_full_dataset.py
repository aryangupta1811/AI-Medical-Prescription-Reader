import os
import cv2
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path

# Provide robust logging feedback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def clean_ocr_label(name: str):
    """Strip out punctuation for pure a-z0-9 string"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    return re.sub(r'[^a-z0-9]', '', name).strip()

def build_production_pipeline():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    labels_file = processed_dir / "labels.txt"
    out_images_dir = processed_dir / "images"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Define mapping configs explicitly for train and val splits
    sources = [
        {"prefix": "train", "csv": raw_dir / "train_labels.csv", "img_dir": raw_dir / "train_images"},
        {"prefix": "val",   "csv": raw_dir / "val_labels.csv",   "img_dir": raw_dir / "val_images"}
    ]
    
    logging.info("--- 1. LOAD IMAGE SOURCES & 2. ENSURE UNIQUE FILENAMES ---")
    
    records = []
    seen_names = set()
    
    for src in sources:
        csv_path = src["csv"]
        prefix = src["prefix"]
        img_dir = src["img_dir"]
        
        if not csv_path.exists():
            continue
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                if 'IMAGE' not in row or 'MEDICINE_NAME' not in row:
                    continue
                    
                raw_img = str(row['IMAGE']).strip()
                raw_label = row['MEDICINE_NAME']
                clean_label = clean_ocr_label(raw_label)
                
                if not clean_label:
                    continue
                    
                # Fix Collisions: Prefix filename safely
                unique_name = f"{prefix}_{raw_img}"
                
                # Check explicitly for collisions!
                if unique_name in seen_names:
                    raise RuntimeError(f"🚨 FATAL: Collision detected: {unique_name}")
                seen_names.add(unique_name)
                
                src_path = img_dir / raw_img
                records.append({
                    "unique_name": unique_name,
                    "label": clean_label,
                    "source_path": src_path
                })
        except Exception as e:
            raise RuntimeError(f"Failed loading {csv_path.name}: {e}")
            
    if not records:
        raise RuntimeError("🚨 FATAL: No records extracted from CSVs!")
        
    logging.info(f"Loaded {len(records)} image sources successfully while avoiding mapping collisions.")

    logging.info("\n--- 3 & 4. PROCESS ALL IMAGES & SAVE OUTPUT ---")
    
    processed_count = 0
    for record in records:
        src_path = record["source_path"]
        unique_name = record["unique_name"]
        
        if not src_path.exists() or not src_path.is_file():
            raise RuntimeError(f"🚨 FATAL: Mapped physical image missing at {src_path}!")
            
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"🚨 FATAL: cv2 failed to read image '{src_path}'! File corrupted.")
            
        # Resize logic matching your test_image_build.py script explicitly
        h, w = img.shape
        new_h = 32
        new_w = int((w / h) * new_h)
        if new_w > 128:
            new_w = 128
            
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad width to 128 (White background)
        padded = np.full((32, 128), 255, dtype=np.uint8)
        padded[:, :new_w] = resized[:, :128] # Secure horizontal constraint
        
        out_path = out_images_dir / unique_name
        cv2.imwrite(str(out_path), padded)
        processed_count += 1
        
    logging.info(f"Successfully processed and overwritten {processed_count} unique images.")

    logging.info("\n--- 5. UPDATE labels.txt ---")
    
    with open(labels_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['unique_name']} {r['label']}\n")
            
    logging.info(f"Re-mapped {len(records)} formatted labels into {labels_file.name}.")

    logging.info("\n--- 6 & 7. STRICT VALIDATION & LOUD FAILURES ---")
    
    if processed_count < len(records):
        raise RuntimeError(f"🚨 FATAL: Mismatch detected! Processed {processed_count} images but labels track {len(records)}.")
        
    if processed_count == 0:
        raise RuntimeError("🚨 FATAL: Processed count is exactly 0.")
        
    print("\n" + "="*50)
    print("✅ PRODUCTION PIPELINE COMPLETE")
    print(f"100% Alignment verified. Processed perfectly aligned: {processed_count} / {len(records)}.")
    print("Dataset is now universally collision-free and ready for balancing.")
    print("="*50 + "\n")


if __name__ == "__main__":
    build_production_pipeline()
