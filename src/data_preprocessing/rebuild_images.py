import os
import cv2
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure strict logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def rebuild_images():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    labels_file = processed_dir / "labels.txt"
    out_images_dir = processed_dir / "images"

    out_images_dir.mkdir(parents=True, exist_ok=True)

    logging.info("--- 1. LOAD SOURCE IMAGES ---")
    
    # Strictly bind image pools to the raw label CSVs
    train_csv = raw_dir / "train_labels.csv"
    val_csv = raw_dir / "val_labels.csv"
    
    csv_valid_images = set()
    for csv_path in [train_csv, val_csv]:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'IMAGE' in df.columns:
                    csv_valid_images.update(df['IMAGE'].astype(str).str.strip().tolist())
            except Exception as e:
                raise RuntimeError(f"🚨 FATAL ERROR: Cannot read CSV {csv_path.name}: {e}")
                
    if not csv_valid_images:
        raise RuntimeError("🚨 FATAL ERROR: No image filenames extracted from train or val CSVs!")

    # Check targets required by labels.txt
    if not labels_file.exists():
        raise RuntimeError(f"🚨 FATAL ERROR: Labels tracker missing! {labels_file}")
        
    labels_txt_images = set()
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels_txt_images.add(parts[0])
                
    if not labels_txt_images:
        raise RuntimeError("🚨 FATAL ERROR: labels.txt is completely empty!")

    logging.info("--- 2 & 3. PROCESS EACH IMAGE AND SAVE ---")
    
    search_dirs = [raw_dir / "train_images", raw_dir / "val_images"]
    processed_count = 0
    skipped_count = 0
    sample_files_saved = []
    
    for img_name in labels_txt_images:
        # Step 1: Only process images that exist in train_labels / val_labels CSVs
        if img_name not in csv_valid_images:
            logging.warning(f"Skipping {img_name} - Not found in raw label CSVs.")
            skipped_count += 1
            continue
            
        src_path = None
        for sdir in search_dirs:
            temp = sdir / img_name
            if temp.exists() and temp.is_file():
                src_path = temp
                break
                
        if not src_path:
            raise RuntimeError(f"🚨 FATAL ERROR: Image '{img_name}' is physically missing from {search_dirs}!")
            
        # Step 2: Read cv2, Grayscale, Resize, Pad
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"🚨 FATAL ERROR: cv2 cannot read '{src_path}'. Found file but data may be corrupted!")
            
        h, w = img.shape
        new_h = 32
        new_w = int(w * (new_h / h))
        
        # Proportional width, explicitly capped if over 128
        if new_w > 128:
            new_w = 128
            
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad to width 128 (white background = 255)
        padded = np.full((32, 128), 255, dtype=np.uint8)
        padded[:, :new_w] = resized
        
        # Step 3: Save Output (safe overwrite via cv2, EXACT filename)
        out_path = out_images_dir / img_name
        cv2.imwrite(str(out_path), padded)
        
        processed_count += 1
        if len(sample_files_saved) < 10:
            sample_files_saved.append(img_name)

    logging.info("--- 4 & 5. DEBUG LOGGING & STRICT VALIDATION ---")
    
    logging.info(f"Target count expected by labels.txt: {len(labels_txt_images)}")
    logging.info(f"Total valid images robustly processed & saved: {processed_count}")
    logging.info(f"Total images skipped: {skipped_count}")
    logging.info(f"Sample exported filenames: {sample_files_saved}")
    
    logging.info("--- 6. FAIL LOUDLY ---")
    if processed_count == 0:
        raise RuntimeError("🚨 FATAL ERROR: Output count is 0. No images were generated!")
        
    if processed_count != len(labels_txt_images):
        raise RuntimeError(f"🚨 FATAL ERROR: Mismatch! We successfully saved {processed_count} images but labels.txt expects exactly {len(labels_txt_images)}!")

    print("\n" + "="*50)
    print("✅ REBUILD SUCCESSFUL: Complete 1-to-1 Image Match achieved.")
    print("="*50 + "\n")

if __name__ == "__main__":
    rebuild_images()
