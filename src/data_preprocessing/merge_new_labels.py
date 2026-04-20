import os
import shutil
import csv

NEW_LABELS = "data/raw/new_labels.csv"
NEW_IMAGES_DIR = "data/raw/ready_to_label"
OUTPUT_LABELS = "data/processed/labels_balanced.txt"
OUTPUT_IMAGES = "data/processed/images_balanced"

def merge_data():
    if not os.path.exists(NEW_LABELS):
        print(f"Error: {NEW_LABELS} not found.")
        return
        
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    
    count = 0
    with open(NEW_LABELS, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip the header row
        except StopIteration:
            pass
            
        with open(OUTPUT_LABELS, "a", encoding="utf-8") as out:
            for row in reader:
                if len(row) < 2: continue
                img_name, label = row[0].strip(), row[1].strip()
                
                if not label or label.lower() == "skip":
                    continue
                    
                src_path = os.path.join(NEW_IMAGES_DIR, img_name)
                dst_path = os.path.join(OUTPUT_IMAGES, img_name)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    # Write in standard format: image_name.jpg label
                    out.write(f"{img_name} {label}\n")
                    count += 1
                    
    print(f"[SUCCESS] Merged {count} newly labeled images directly into the core training hub!")

if __name__ == "__main__":
    merge_data()
