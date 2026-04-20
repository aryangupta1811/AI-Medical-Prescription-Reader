import os
import random
import shutil
import logging
from pathlib import Path
from collections import Counter, defaultdict

# Configure logging to prevent silent failures
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def balance_dataset(max_samples_per_label=20):
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "processed"
    labels_file = processed_dir / "labels.txt"
    balanced_labels_file = processed_dir / "labels_balanced.txt"
    images_dir = processed_dir / "images"
    balanced_images_dir = processed_dir / "images_balanced"

    if not labels_file.exists():
        logging.error(f"Cannot find {labels_file}. Run prep pipeline first.")
        raise FileNotFoundError(f"Missing labels file: {labels_file}")

    logging.info("--- STEP 1: ANALYZE DISTRIBUTION ---")
    
    label_to_images = defaultdict(list)
    total_raw_samples = 0
    
    # Read raw labels securely mapped to their images
    with open(labels_file, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                img_name = parts[0]
                label = " ".join(parts[1:])
                label_to_images[label].append(img_name)
                total_raw_samples += 1
            else:
                logging.warning(f"Skipping malformed line {index+1}: '{line.strip()}'")

    if total_raw_samples == 0:
        raise ValueError("No valid samples found in labels.txt!")

    label_counts = {label: len(imgs) for label, imgs in label_to_images.items()}
    unique_labels = len(label_counts)
    counter = Counter(label_counts)
    
    print("\n" + "="*50)
    print("📊 DISTRIBUTION (BEFORE BALANCING)")
    print("="*50)
    print(f"Total Samples: {total_raw_samples}")
    print(f"Total Unique Labels: {unique_labels}")
    print("\nTop 10 Most Frequent Labels:")
    for label, count in counter.most_common(10):
        print(f"  - {label}: {count} samples")

    logging.info(f"\n--- STEP 2: LIMIT SAMPLES PER LABEL (MAX {max_samples_per_label}) ---")
    balanced_records = []
    
    for label, img_list in label_to_images.items():
        # Step 6 prep check: ensure no string is literally entirely whitespaces/empty
        if not label.strip():
            logging.warning(f"Found empty label mapped to images: {img_list}. Discarding.")
            continue
            
        if len(img_list) > max_samples_per_label:
            # Randomly undersample
            selected_images = random.sample(img_list, max_samples_per_label)
        else:
            selected_images = img_list
            
        for img in selected_images:
            balanced_records.append((img, label))
            
    # Optional: shuffle the whole dataset to prevent sequence bias
    random.shuffle(balanced_records)
            
    logging.info(f"Selected {len(balanced_records)} balanced samples.")

    logging.info("\n--- STEP 3: CREATE BALANCED DATASET ---")
    
    with open(balanced_labels_file, "w", encoding="utf-8") as f:
        for img, label in balanced_records:
            f.write(f"{img} {label}\n")
            
    logging.info(f"Saved balanced labels to {balanced_labels_file.name}")

    logging.info("\n--- STEP 4: OPTIONAL IMAGE FILTERING ---")
    
    balanced_images_dir.mkdir(parents=True, exist_ok=True)
    missing_images = []
    copied_count = 0
    
    for img_name, label in balanced_records:
        src_img_path = images_dir / img_name
        dest_img_path = balanced_images_dir / img_name
        
        if src_img_path.exists():
            if not dest_img_path.exists():
                shutil.copy2(src_img_path, dest_img_path)
            copied_count += 1
        else:
            missing_images.append(img_name)
            
    logging.info(f"Copied {copied_count} balanced images over to {balanced_images_dir.name}/")
    
    logging.info("\n--- STEP 5: PRINT AFTER STATS ---")
    
    balanced_counter = Counter([label for img, label in balanced_records])
    
    print("\n" + "="*50)
    print("⚖️ DISTRIBUTION (AFTER BALANCING)")
    print("="*50)
    print(f"Total Samples: {len(balanced_records)}")
    print(f"Total Unique Labels: {len(balanced_counter)}")
    print("\nTop 10 Labels (Should be clamped/balanced):")
    for label, count in balanced_counter.most_common(10):
        print(f"  - {label}: {count} samples")

    logging.info("\n--- STEP 6: SANITY CHECK ---")
    
    if len(balanced_records) == 0:
        raise RuntimeError("🚨 FATAL ERROR: Balanced dataset has 0 elements! Nothing was saved.")
        
    empty_labels = [img for img, label in balanced_records if not label.strip()]
    if empty_labels:
        raise RuntimeError(f"🚨 FATAL ERROR: Found empty labels written! Samples: {empty_labels[:5]}")
        
    if missing_images:
        logging.error(f"🚨 FATAL ERROR: {len(missing_images)} chosen images are missing from {images_dir.name}!")
        logging.error(f"Sample missing: {missing_images[:5]}")
        raise FileNotFoundError("Image paths missing. They must exist in data/processed/images/ before copying.")
        
    logging.info("✅ Sanity check passed: All balanced images logically exist and labels are valid!")

if __name__ == "__main__":
    # Feel free to adjust max_samples_per_label parameter here
    balance_dataset(max_samples_per_label=20)
