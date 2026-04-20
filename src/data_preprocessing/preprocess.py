import os
import re
import cv2
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import random

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class MedicalDataPreprocessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.raw_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.img_out_dir = self.processed_dir / "images"
        self.labels_out_txt = self.processed_dir / "labels.txt"
        self.dict_out_json = self.processed_dir / "medicine_dictionary_clean.json"
        self.generic_out_json = self.processed_dir / "generic_to_brand.json"

        # Ensure processed directories exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.img_out_dir.mkdir(parents=True, exist_ok=True)

    def standardize_file_structure(self):
        logging.info("--- STEP 1: STANDARDIZE FILE STRUCTURE ---")
        renames = [
            ("training_words", "train_images"),
            ("validation_words", "val_images"),
            ("testing_words", "test_images"),
            ("Medicine_Details.csv", "medicine_dictionary.csv")
        ]
        
        for old, new in renames:
            old_path = self.raw_dir / old
            new_path = self.raw_dir / new
            if old_path.exists() and not new_path.exists():
                old_path.rename(new_path)
                logging.info(f"Renamed: {old} -> {new}")
            elif not old_path.exists() and not new_path.exists():
                logging.warning(f"Could not find starting path: {old_path} (it may have already been renamed or doesn't exist).")
                
        logging.info("File structure standard verified.")

    def _clean_ocr_label(self, name):
        """Convert to lowercase, keep a-z 0-9 only"""
        if pd.isna(name):
            return ""
        name = str(name).lower()
        name = re.sub(r'[^a-z0-9]', '', name)
        return name.strip()

    def clean_ocr_labels(self):
        logging.info("--- STEP 2: CLEAN OCR LABELS ---")
        train_csv = self.raw_dir / "train_labels.csv"
        val_csv = self.raw_dir / "val_labels.csv"
        
        df_list = []
        for csv_path in [train_csv, val_csv]:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df_list.append(df)
                except Exception as e:
                    logging.error(f"Error reading {csv_path}: {e}")
            else:
                logging.warning(f"File missing: {csv_path}")
                
        if not df_list:
            logging.error("No label CSVs found! Cannot process labels.")
            return []

        combined_df = pd.concat(df_list, ignore_index=True)
        
        valid_records = []
        for _, row in combined_df.iterrows():
            if 'IMAGE' not in row or 'MEDICINE_NAME' not in row:
                continue
                
            img_filename = str(row['IMAGE']).strip()
            raw_medicine = row['MEDICINE_NAME']
            
            clean_label = self._clean_ocr_label(raw_medicine)
            
            if clean_label and img_filename:
                # Store valid record
                valid_records.append((img_filename, clean_label))
        
        # Write to labels.txt
        with open(self.labels_out_txt, 'w', encoding='utf-8') as f:
            for img_name, label in valid_records:
                f.write(f"{img_name} {label}\n")
                
        logging.info(f"Processed and cleaned {len(valid_records)} OCR labels. Saved to labels.txt")
        return valid_records

    def copy_and_preprocess_images(self, valid_records):
        logging.info("--- STEP 3: COPY + PREPROCESS IMAGES ---")
        
        # Determine expected valid images
        expected_images = {img for img, _ in valid_records}
        missing_images = []
        
        search_dirs = [self.raw_dir / "train_images", self.raw_dir / "val_images"]
        
        processed_count = 0
        for img_name in expected_images:
            source_path = None
            for s_dir in search_dirs:
                temp_path = s_dir / img_name
                if temp_path.exists() and temp_path.is_file():
                    source_path = temp_path
                    break
            
            if not source_path:
                missing_images.append(img_name)
                continue
                
            # Read Image
            img = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                missing_images.append(img_name)
                continue
                
            # Resize height = 32 (maintain aspect ratio)
            h, w = img.shape
            new_h = 32
            new_w = int(w * (new_h / h))
            
            # Capping width to 128 if necessary
            if new_w > 128:
                new_w = 128
                
            resized = cv2.resize(img, (new_w, new_h))
            
            # Pad width to 128 (padding with white/255)
            padded = np.zeros((32, 128), dtype=np.uint8)
            padded.fill(255)
            padded[:, :new_w] = resized
            
            # Save into data/processed/images/
            output_path = self.img_out_dir / img_name
            # NOTE: Normalization (0-1) is generally handled via explicit transforms in 
            # Torch Datasets (e.g. ToTensor) dynamically instead of saving unreadable image formats.
            # However, for pure image pre-processing, we structure the pixels 0-255 cleanly.
            cv2.imwrite(str(output_path), padded)
            processed_count += 1
            
        logging.info(f"Successfully preprocessed {processed_count} images.")
        return missing_images

    def _clean_dict_medicine_name(self, name):
        if pd.isna(name): return ""
        name = str(name).lower()
        # Remove dosage (mg, ml, g, mcg)
        name = re.sub(r'\b\d+(\.\d+)?\s*(mg|ml|g|mcg)\b', ' ', name)
        # Remove words
        name = re.sub(r'\b(tablet|syrup|injection|capsule)s?\b', ' ', name)
        # Remove punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        # Remove extra spaces safely
        return re.sub(r'\s+', ' ', name).strip()

    def clean_medicine_dictionary(self):
        logging.info("--- STEP 4: CLEAN MEDICINE DICTIONARY ---")
        dict_csv = self.raw_dir / "medicine_dictionary.csv"
        
        if not dict_csv.exists():
            logging.error(f"Dictionary CSV not found at {dict_csv}")
            return
            
        try:
            df = pd.read_csv(dict_csv)
        except Exception as e:
            logging.error(f"Failed to read {dict_csv}: {e}")
            return
            
        keep_columns = [
            'medicine_name', 'composition', 'uses', 'side_effects',
            'excellent_review_percent', 'average_review_percent', 'poor_review_percent'
        ]
        
        # Verify requested columns exist
        available_cols = [c for c in keep_columns if c in df.columns]
        if 'medicine_name' not in available_cols:
            logging.error("'medicine_name' column not found in dictionary!")
            return
            
        df = df[available_cols].copy()
        
        clean_data = []
        for _, row in df.iterrows():
            original_name = row['medicine_name']
            cleaned_name = self._clean_dict_medicine_name(original_name)
            
            if cleaned_name:
                record = {col: str(row[col]) if pd.notnull(row[col]) else "" for col in available_cols}
                record['original_name'] = original_name
                record['medicine_name'] = cleaned_name # Replace with clean name
                clean_data.append(record)
                
        with open(self.dict_out_json, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=4)
            
        logging.info(f"Cleaned {len(clean_data)} valid medicine dictionary records.")

    def _clean_generic_name(self, name):
        if pd.isna(name): return ""
        name = str(name).lower()
        name = re.sub(r'[^\w\s]', ' ', name)
        return re.sub(r'\s+', ' ', name).strip()

    def build_generic_brand_mapping(self):
        logging.info("--- STEP 5: GENERIC → BRAND MAPPING ---")
        train_csv = self.raw_dir / "train_labels.csv"
        val_csv = self.raw_dir / "val_labels.csv"
        
        df_list = []
        for csv_path in [train_csv, val_csv]:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df_list.append(df)
                
        if not df_list:
            return
            
        combined_df = pd.concat(df_list, ignore_index=True)
        
        mapping = {}
        if 'GENERIC_NAME' not in combined_df.columns or 'MEDICINE_NAME' not in combined_df.columns:
            logging.warning("GENERIC_NAME or MEDICINE_NAME column missing from labels.")
            return
            
        for _, row in combined_df.iterrows():
            generic_raw = row['GENERIC_NAME']
            brand_raw = row['MEDICINE_NAME']
            
            clean_generic = self._clean_generic_name(generic_raw)
            clean_brand = self._clean_ocr_label(brand_raw)
            
            if clean_generic and clean_brand:
                if clean_generic not in mapping:
                    mapping[clean_generic] = set()
                mapping[clean_generic].add(clean_brand)
                
        # Convert sets to lists
        mapping = {k: list(v) for k, v in mapping.items()}
        
        with open(self.generic_out_json, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4)
            
        logging.info(f"Created mapping for {len(mapping)} generic compounds.")

    def run_validations(self, valid_records, missing_images):
        logging.info("--- STEP 6: VALIDATION CHECKS ---")
        total_samples = len(valid_records)
        labels = [label for _, label in valid_records]
        unique_labels = len(set(labels))
        
        counter = Counter(labels)
        top_10 = counter.most_common(10)
        
        empty_labels = [img for img, label in valid_records if not label.strip()]
        
        print("\n" + "="*40)
        print("📊 PIPELINE VALIDATION REPORT")
        print("="*40)
        print(f"🔹 Total Clean Samples: {total_samples}")
        print(f"🔹 Unique Labels Count: {unique_labels}")
        print("🔹 Top 10 Most Frequent Labels:")
        for label, count in top_10:
            print(f"    - {label}: {count}")
            
        print(f"🔹 Missing Images: {len(missing_images)}")
        if missing_images:
            print(f"    Examples: {missing_images[:5]}...")
        
        print(f"🔹 Empty/Invalid Labels from parsed CSV: {len(empty_labels)}")
        print("="*40 + "\n")

    def run_quality_check(self):
        logging.info("--- STEP 7: OUTPUT QUALITY CHECK ---")
        
        if not self.labels_out_txt.exists():
            logging.error("labels.txt not found. Can't run quality check.")
            return
            
        with open(self.labels_out_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if not lines:
            logging.error("labels.txt is empty.")
            return
            
        samples = random.sample(lines, min(10, len(lines)))
        
        plt.figure(figsize=(15, 6))
        
        valid_plots = 0
        for i, line in enumerate(samples):
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                label = " ".join(parts[1:])
                
                img_path = self.img_out_dir / img_name
                if img_path.exists():
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        plt.subplot(2, 5, valid_plots + 1)
                        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                        plt.title(f"Label: {label}")
                        plt.axis('off')
                        valid_plots += 1
                        
        if valid_plots > 0:
            logging.info(f"Displaying {valid_plots} quality check samples.")
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("No valid images found to display for quality check.")

    def run_pipeline(self):
        logging.info("🚀 Starting Preprocessing Pipeline...")
        self.standardize_file_structure()
        valid_records = self.clean_ocr_labels()
        
        missing_images = []
        if valid_records:
            missing_images = self.copy_and_preprocess_images(valid_records)
            
        self.clean_medicine_dictionary()
        self.build_generic_brand_mapping()
        
        self.run_validations(valid_records, missing_images)
        self.run_quality_check()
        logging.info("✅ Preprocessing Complete!")


if __name__ == "__main__":
    # Adjust base_dir depending on where you run this script from
    # E.g. running from root `python src/data_preprocessing/preprocess.py`
    base_proj_dir = Path(__file__).resolve().parent.parent.parent
    
    pipeline = MedicalDataPreprocessor(base_proj_dir)
    pipeline.run_pipeline()
