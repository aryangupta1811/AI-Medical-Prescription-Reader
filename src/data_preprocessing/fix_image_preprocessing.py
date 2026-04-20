import os
import cv2
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_image_preprocessing():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    labels_file = processed_dir / "labels.txt"
    out_images_dir = processed_dir / "images"

    out_images_dir.mkdir(parents=True, exist_ok=True)

    logging.info("--- STEP 1: LOAD IMAGE PATHS ---")
    
    # Extract "valid" filenames from the labels CSV source
    train_csv = raw_dir / "train_labels.csv"
    val_csv = raw_dir / "val_labels.csv"
    
    csv_valid_images = set()
    for csv_path in [train_csv, val_csv]:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'IMAGE' in df.columns:
                    # Clean filename strings strictly
                    csv_valid_images.update(df['IMAGE'].astype(str).str.strip().tolist())
            except Exception as e:
                logging.error(f"Failed to read {csv_path.name}: {e}")
                raise RuntimeError(f"🚨 FATAL: CSV load failed: {e}")
                
    if not csv_valid_images:
        raise RuntimeError("🚨 FATAL ERROR: No image filenames extracted from CSVs!")
        
    logging.info(f"Determined {len(csv_valid_images)} total valid image filenames strictly from CSVs.")

    # Load the exact expectations from labels.txt
    if not labels_file.exists():
        raise RuntimeError(f"🚨 FATAL ERROR: {labels_file.name} does not exist!")
        
    expected_images = set()
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                expected_images.add(parts[0])
                
    if not expected_images:
        raise RuntimeError("🚨 FATAL ERROR: labels.txt is completely empty!")
        
    logging.info(f"Target batch: {len(expected_images)} unique images expected by labels.txt.")

    logging.info("\n--- STEP 2 & 3: COPY + PREPROCESS IMAGES ---")
    
    search_dirs = [raw_dir / "train_images", raw_dir / "val_images"]
    processed_count = 0
    sample_files_saved = []

    for img_name in expected_images:
        # Step 5 check: Only process if it's considered valid from the CSVs
        if img_name not in csv_valid_images:
            raise RuntimeError(f"🚨 FATAL ERROR: labels.txt demands '{img_name}' but it's not present in your CSVs!")

        # 1. Locate physical source file
        src_path = None
        for sdir in search_dirs:
            temp = sdir / img_name
            if temp.exists() and temp.is_file():
                src_path = temp
                break
                
        if not src_path:
            raise RuntimeError(f"🚨 FATAL ERROR: File missing! Could not physically find '{img_name}' in raw folders.")
            
        # 2. cv2 Load
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"🚨 FATAL ERROR: cv2 cannot read image data from {src_path}. File might be corrupted.")
            
        # 3. Resize & Pad
        h, w = img.shape
        new_h = 32
        new_w = int(w * (new_h / h))
        
        # Cap width
        if new_w > 128:
            new_w = 128
            
        resized = cv2.resize(img, (new_w, new_h))
        
        # Pad width to 128 (White background = 255)
        padded = np.full((32, 128), 255, dtype=np.uint8)
        padded[:, :new_w] = resized
        
        # 4. Save EXACT filename
        out_path = out_images_dir / img_name
        cv2.imwrite(str(out_path), padded)
        
        processed_count += 1
        if len(sample_files_saved) < 5:
            sample_files_saved.append(img_name)

    logging.info("\n--- STEP 4 & 5: VALIDATION & FAIL LOUDLY ---")
    
    logging.info(f"Total processed images securely saved: {processed_count}")
    logging.info(f"Sample filenames exported: {sample_files_saved}")
    
    if processed_count == 0:
        raise RuntimeError("🚨 FATAL ERROR: 0 images were processed and saved!")
        
    if processed_count != len(expected_images):
        raise RuntimeError(f"🚨 FATAL ERROR: Mismatched counts! Preprocessed {processed_count} but labels.txt demands {len(expected_images)}.")

    print("\n" + "="*50)
    print("✅ IMAGE PREPROCESSING FIX COMPLETE!")
    print(f"Every label in {labels_file.name} now securely maps to a real, padded .png image inside {out_images_dir.name}/.")
    print("="*50 + "\n")

if __name__ == "__main__":
    fix_image_preprocessing()
