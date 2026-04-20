import os
import random
import uuid
import json
import numpy as np
import cv2
import urllib.request
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# CONFIGURATION
# -----------------------------
FONTS_DIR = "data/processed/fonts"
OUTPUT_IMAGES_DIR = "data/processed/images"
OUTPUT_LABELS_FILE = "data/processed/labels.txt"

KNOWLEDGE_BASE_PATH = "data/processed/final_knowledge_base.json"

NUM_IMAGES_TO_GENERATE = 5000
IMAGE_SIZE = (128, 32)

# Download Google Fonts natively
FONT_URLS = {
    "DancingScript": "https://raw.githubusercontent.com/google/fonts/main/ofl/dancingscript/DancingScript%5Bwght%5D.ttf",
    "Caveat": "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",
    "GreatVibes": "https://raw.githubusercontent.com/google/fonts/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    "Pacifico": "https://raw.githubusercontent.com/google/fonts/main/ofl/pacifico/Pacifico-Regular.ttf",
    "ShadowsIntoLight": "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight-Regular.ttf",
    "AlexBrush": "https://raw.githubusercontent.com/google/fonts/main/ofl/alexbrush/AlexBrush-Regular.ttf",
    "Allura": "https://raw.githubusercontent.com/google/fonts/main/ofl/allura/Allura-Regular.ttf"
}

# -----------------------------
# STEP 1: FONT MANAGEMENT
# -----------------------------
def download_fonts():
    os.makedirs(FONTS_DIR, exist_ok=True)
    font_paths = []

    for name, url in FONT_URLS.items():
        save_path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if not os.path.exists(save_path):
            print(f"[DOWNLOAD] Fetching {name}.ttf...")
            try:
                urllib.request.urlretrieve(url, save_path)
            except Exception as e:
                print(f"[ERROR] Failed to download {name}: {e}")
                continue
        font_paths.append(save_path)

    return [f for f in font_paths if os.path.exists(f)]

# -----------------------------
# STEP 2: DISTORTION ENGINE
# -----------------------------
def apply_distortions(img):
    cv_img = np.array(img)

    # 1. Random Gaussian Blur
    if random.random() > 0.4:
        ksize = random.choice([(3,3), (5,5)])
        cv_img = cv2.GaussianBlur(cv_img, ksize, random.uniform(0.5, 1.5))

    # 2. Random Affine Warp (Shear)
    if random.random() > 0.5:
        h, w = cv_img.shape
        shear_factor = random.uniform(-0.3, 0.3)
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        cv_img = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. Additive Gaussian Noise (speckle)
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), cv_img.shape).astype(np.float32)
        cv_img = np.clip(cv_img + noise, 0, 255).astype(np.uint8)

    # 4. Invert colors randomly
    if random.random() > 0.9:
        cv_img = 255 - cv_img

    return Image.fromarray(cv_img)

# -----------------------------
# STEP 3: GENERATOR LOGIC
# -----------------------------
def generate_images():
    print("\n[INFO] Starting Synthetic Data Generator...")
    
    fonts = download_fonts()
    if not fonts:
        print("[ERROR] No fonts downloaded. Aborting.")
        return

    # Load Knowledge Base for terms
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"[ERROR] Knowledge Base not found at {KNOWLEDGE_BASE_PATH}")
        return

    with open(KNOWLEDGE_BASE_PATH, "r") as f:
        kb = json.load(f)
    
    medicine_names = list(kb.keys())
    if not medicine_names:
        print("[ERROR] Knowledge Base is empty.")
        return

    # To add noise to the strings, we might misspell slightly or crop, but keeping the ground truth pure is safer for CRNN
    valid_chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ")

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    
    generated_count = 0
    new_labels = []

    print(f"\n[INFO] Generating {NUM_IMAGES_TO_GENERATE} Cursive Images. Buckle up...")

    for _ in range(NUM_IMAGES_TO_GENERATE):
        drug = random.choice(medicine_names)
        
        # Clean string to match allowed chars
        drug = "".join([c for c in drug.lower() if c in valid_chars])
        
        if not drug.strip(): continue

        # Sometimes add mock dosages
        if random.random() > 0.7:
             drug = f"{drug} {random.choice(['100mg', '200mg', '500mg', '10ml', '0.5mg'])}"

        # Select font & render
        font_path = random.choice(fonts)
        font_size = random.randint(20, 32)
        font = ImageFont.truetype(font_path, font_size)

        # Create blank canvas (noisy white/gray background)
        bg_color = random.randint(200, 255)
        img = Image.new('L', IMAGE_SIZE, color=bg_color)
        draw = ImageDraw.Draw(img)

        # Dynamic positioning
        # We need the drug text to roughly fit the bounds
        try:
            # Pillow >= 8.0 support
            bbox = draw.textbbox((0,0), drug, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(drug, font=font)

        # If it's way too wide, squash it or just accept clipping
        # CRNN can learn squashed/clipped 
        x = random.randint(-5, max(0, IMAGE_SIZE[0] - text_w))
        y = random.randint(-5, max(0, IMAGE_SIZE[1] - text_h - 10))

        text_color = random.randint(0, 100) # dark gray/black pen

        draw.text((x, y), drug, fill=text_color, font=font)

        # Distort the image to emulate a real scan
        img = apply_distortions(img)

        filename = f"synth_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
        
        img.save(save_path)
        new_labels.append(f"{filename}\t{drug}\n")
        
        generated_count += 1

    print(f"\n[SUCCESS] Successfully minted {generated_count} synthetic images!")

    # Append to labels.txt
    print(f"[INFO] Merging {generated_count} labels into {OUTPUT_LABELS_FILE}...")
    with open(OUTPUT_LABELS_FILE, "a", encoding="utf-8") as f:
        f.writelines(new_labels)
        
    print("[SUCCESS] Data Injection Complete. Ready for Dataset Balancing.")

if __name__ == "__main__":
    generate_images()
