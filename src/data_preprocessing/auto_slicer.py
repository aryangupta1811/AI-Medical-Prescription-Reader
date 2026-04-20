import os
import sys
import cv2

# Allow the script to resolve 'src' from the root directory mathematically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.pipeline.line_processing import extract_ocr_data, group_into_lines

RAW_DIR = "data/raw/unlabeled_crops"
OUTPUT_DIR = "data/raw/ready_to_label"

def slice_images_into_crops():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {RAW_DIR}")
        return

    crop_id = 0
    for img_name in image_files:
        img_path = os.path.join(RAW_DIR, img_name)
        print(f"[*] Analyzing and Slicing: {img_name}")
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        # Use our AI pipeline to find the text boxes
        ocr_data = extract_ocr_data(img_path)
        lines = group_into_lines(ocr_data)
        
        for line in lines:
            if not line: continue
            
            # Draw a box around the entire horizontal sentence
            x_min = int(min(w['x_min'] for w in line))
            x_max = int(max(w['x_max'] for w in line))
            y_min = int(min(w['y_min'] for w in line))
            y_max = int(max(w['y_max'] for w in line))
            
            # Add a 5 pixel padding so the letters don't get cut off
            pad = 5
            h, w_img, _ = img.shape
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w_img, x_max + pad)
            y_max = min(h, y_max + pad)
            
            crop = img[y_min:y_max, x_min:x_max]
            
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 10: 
                continue
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"crop_{crop_id}.jpg"), crop)
            crop_id += 1

    print(f"\n[SUCCESS] Sliced your block images into {crop_id} perfect single-line crops!")
    print(f"Saved into: {OUTPUT_DIR}")

if __name__ == "__main__":
    slice_images_into_crops()
