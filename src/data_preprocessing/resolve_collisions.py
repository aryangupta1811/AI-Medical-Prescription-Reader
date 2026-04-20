import os
import cv2
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path

# Provide clear, deterministic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def _clean_ocr_label(name: str):
    """Strip out punctuation and spaces strictly"""
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name.strip()

def resolve_dataset_collisions():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    labels_file = processed_dir / "labels.txt"
    out_images_dir = processed_dir / "images"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration dict explicitly mapping origins
    sources = [
        {"prefix": "train", "csv": raw_dir / "train_labels.csv", "img_dir": raw_dir / "train_images"},
        {"prefix": "val",   "csv": raw_dir / "val_labels.csv",   "img_dir": raw_dir / "val_images"}
    ]
    
    logging.info("--- 1. LOAD IMAGE SOURCES & 2. CREATE UNIQUE NAMES ---")
    
    valid_records = []
    seen_unique_names = set()
    
    for src in sources:
        csv_path = src["csv"]
        prefix = src["prefix"]
        img_dir = src["img_dir"]
        
        if not csv_path.exists():
            logging.warning(f"File {csv_path.name} not found! Skipping.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                if 'IMAGE' not in row or 'MEDICINE_NAME' not in row:
                    continue
                    
                raw_img_name = str(row['IMAGE']).strip()
                raw_label = row['MEDICINE_NAME']
                
                # We also need to map the clean label format expected by the model
                clean_label = _clean_ocr_label(raw_label)
                if not clean_label:
                    continue
                    
                # Fix Collisions: dynamically prefix the filename (train_0.png)
                unique_img_name = f"{prefix}_{raw_img_name}"
                
                # Check explicitly for newly formed collisions!
                if unique_img_name in seen_unique_names:
                    raise RuntimeError(f"🚨 FATAL: Duplicate filename detected even after prefixing: {unique_img_name}")
                    
                seen_unique_names.add(unique_img_name)
                
                raw_img_path = img_dir / raw_img_name
                valid_records.append({
                    "unique_name": unique_img_name,
                    "label": clean_label,
                    "source_path": raw_img_path
                })
        except Exception as e:
            raise RuntimeError(f"Failed to extract records from {csv_path.name}: {e}")
            
    if len(valid_records) == 0:
        raise RuntimeError("🚨 FATAL: No valid records compiled!")

    logging.info(f"Generated {len(valid_records)} unique prefixed filename maps.")

    logging.info("\n--- 4 & 5. PROCESS IMAGES & SAVE ---")
    
    processed_count = 0
    for record in valid_records:
        src_path = record["source_path"]
        unique_name = record["unique_name"]
        
        # Immediate Fail Check for Physical Alignment
        if not src_path.exists() or not src_path.is_file():
            raise RuntimeError(f"🚨 FATAL: Physical image missing! Label dictates {src_path.name} should exist at {src_path}")
            
        # Standardize using cv2
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"🚨 FATAL: cv2 failed to decode '{src_path}'! Corrupted data.")
            
        h, w = img.shape
        new_h = 32
        new_w = int(w * (new_h / h))
        if new_w > 128:
            new_w = 128
            
        resized = cv2.resize(img, (new_w, new_h))
        padded = np.full((32, 128), 255, dtype=np.uint8) # 255 = explicit white background
        padded[:, :new_w] = resized
        
        # Save output using unique namespace mapping
        out_path = out_images_dir / unique_name
        cv2.imwrite(str(out_path), padded)
        
        processed_count += 1
        
    logging.info(f"Processed and padded exactly {processed_count} unique image files.")

    logging.info("\n--- 3 & 6. UPDATE AND REGENERATE labels.txt ---")
    
    with open(labels_file, "w", encoding="utf-8") as f:
        for record in valid_records:
            f.write(f"{record['unique_name']} {record['label']}\n")
            
    logging.info(f"Safely rewrote unified labels.txt containing only dynamically prefixed names.")

    logging.info("\n--- 7. ALIGNMENT & FINAL VALIDATION ---")
    
    # Check 1: Mapped matches physically processed
    if processed_count != len(valid_records):
        raise RuntimeError(f"🚨 FATAL: Physical Pipeline Mismatch! We identified {len(valid_records)} labels but processed {processed_count} images.")
    # Check 2: All filenames stored uniquely without duplicates (Handled intrinsically by set)
        
    print("\n" + "="*50)
    print("✅ DATASET COLLISION FIX COMPLETE: 100% Secure File Mappings Established.")
    print("="*50 + "\n")


if __name__ == "__main__":
    resolve_dataset_collisions()
