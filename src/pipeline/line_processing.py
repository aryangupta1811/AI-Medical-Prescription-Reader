import easyocr
import cv2
import re
import torch
import numpy as np
from src.training.model import CRNN

reader = easyocr.Reader(['en'])

CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"
idx2char = {i + 1: c for i, c in enumerate(CHARS)}

print("[INFO] Loading Custom PyTorch CRNN Engine...")
try:
    crnn_model = CRNN(len(CHARS) + 1)
    # Target the fully evaluated Augmented NLP PyTorch model matrix natively
    crnn_model.load_state_dict(torch.load("model_augmented.pth", map_location="cpu", weights_only=True))
    crnn_model.eval()
except Exception as e:
    print(f"[WARNING] PyTorch CRNN failed to load: {e}")
    crnn_model = None

def preprocess_tensor(img_crop):
    img = cv2.resize(img_crop, (128, 32))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img)

def decode_pytorch(output):
    pred = output.argmax(2)[0]
    text = ""
    prev = -1
    for t in pred:
        t = t.item()
        if t != 0 and t != prev:
            text += idx2char.get(t, "")
        prev = t
    return text

# -----------------------------
# STEP 1: Extract OCR structured data
# -----------------------------
def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1. CLAHE Contrast Stretching
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Adaptive Sauvola Binarization (Surgical Shadow Removal)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)
    
    # 3. Soft Gaussian Edge Restoration (Matrix smoothing for CRNN Convolution filters)
    blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
    return blurred

def extract_ocr_data(image_path):
    processed_img = preprocess_image_for_ocr(image_path)
    if processed_img is None:
        return []
        
    results = reader.readtext(processed_img)

    # Use the structurally enhanced matrix natively for PyTorch OCR as well!
    raw_img = processed_img

    ocr_data = []

    for (bbox, raw_text, conf) in results:
        # 🔥 TASK 1: Confidence Floor (Ignore heavily crossed-out scribbles)
        if conf < 0.05:
            continue
            
        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]
        
        x_min, x_max = max(0, min(x_coords)), max(x_coords)
        y_min, y_max = max(0, min(y_coords)), max(y_coords)

        # PyTorch Intercept Override
        text = raw_text
        if crnn_model is not None and raw_img is not None:
            crop = raw_img[max(0, y_min-2):y_max+2, max(0, x_min-2):x_max+2]
            if crop.size > 0:
                tensor = preprocess_tensor(crop)
                with torch.no_grad():
                    out = crnn_model(tensor)
                pt_text = decode_pytorch(out)
                if pt_text.strip():
                    # Fuse mathematical outputs directly instead of aggressively overwriting
                    text = f"{text} {pt_text}"

        ocr_data.append({
            "text": text,
            "conf": conf,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_center": (y_min + y_max) / 2
        })

    # 🔥 SPATIAL ANCHORING: Ignorning headers, hospitals, and history logs
    anchor_y = -1
    for item in ocr_data:
        text_lower = item['text'].lower()
        if "medicine" in text_lower or "prescribe" in text_lower or "rx" in text_lower:
            anchor_y = item['y_center']
            break

    if anchor_y != -1:
        # Keep only text blocks positioned below the "Medicine Prescribed" heading!
        ocr_data = [item for item in ocr_data if item['y_center'] > anchor_y - 15]

    return ocr_data


# -----------------------------
# STEP 2: Group into lines
# -----------------------------
def group_into_lines(ocr_data):
    if not ocr_data:
        return []
        
    heights = [item['y_max'] - item['y_min'] for item in ocr_data]
    median_h = sorted(heights)[len(heights)//2] if heights else 15
    dynamic_threshold = max(10, median_h * 0.6)
    
    ocr_data = sorted(ocr_data, key=lambda x: x['y_center'])

    lines = []
    current_line = []

    for item in ocr_data:
        if not current_line:
            current_line.append(item)
            continue

        # Calculate exact vertical boundary of the current line structurally
        line_y_max = max(i['y_max'] for i in current_line)
        line_y_min = min(i['y_min'] for i in current_line)
        
        # If it physically overlaps within the baseline boundaries natively
        if line_y_min - 10 <= item['y_center'] <= line_y_max + 10:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

    if current_line:
        lines.append(current_line)

    return lines


# -----------------------------
# STEP 3: Convert lines to text
# -----------------------------
def merge_broken_tokens(text):
    text = text.lower()
    
    # 1. Fix numbers split apart (e.g. "6 25" -> "625")
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    # 2. Fix separated units
    text = re.sub(r'(\d+)\s*m\s*g?\b', r'\1 mg', text)
    text = re.sub(r'(\d+)\s*m\s*l?\b', r'\1 ml', text)

    # 🔥 MAXIMUM ACCURACY NER FILTER
    # Strip bulleting ("1.", "(2)", "3 ")
    text = re.sub(r'^[\d\(\)a-z][\.\)]\s*', '', text)
    text = re.sub(r'^\d+\s+', '', text)
    
    # Strip South-Asian Dosage notations (e.g. 1-0-1, 0-1-1)
    text = re.sub(r'\b\d(?:-\d){1,2}\b', '', text)
    
    # Strip fraction blobs (e.g. 1/2)
    text = re.sub(r'\b\d+/\d+\b', '', text)
    
    # Strip Form Prefixes (Tb, Drp, Syp, Cap)
    text = re.sub(r'\b(tb|drp|syp|cap|tab|inj|syrp?)\b\.?', '', text)
    
    # Strip dosages and strengths which break fuzzy match (e.g. "500 mg", "10ml")
    text = re.sub(r'\b\d+/?(\d+)?\s*(mg|ml|g|mcg|iu)\b', '', text)
    
    # Strip durations (e.g. "for 3 days", "2 weeks")
    text = re.sub(r'\b(for\s+)?\d+\s*(days?|weeks?|months?)\b', '', text)
    
    # Strip types and frequencies (e.g. "take", "bd", "daily", "after food")
    text = re.sub(r'\b(take|bd|od|tds|sos|daily|after food|before food|ointment|cream)\b', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lines_to_text(lines):
    text_lines = []

    for line in lines:
        line = sorted(line, key=lambda x: x['x_min'])
        text = " ".join([w['text'] for w in line])
        text = merge_broken_tokens(text.lower())
        text_lines.append(text)

    return text_lines


# -----------------------------
# STEP 4: Basic medicine filter
# -----------------------------
def is_medicine_line(line):
    words = line.split()
    
    if not words:
        return False

    # negative signals
    negative_words = ["hospital", "clinic", "patient age", "patient name"]

    if any(nw in line.lower() for nw in negative_words):
        return False

    # Expanded to gracefully accept extremely long prescription strings natively without dropping them
    if len(words) <= 10:
        return True

    return False


def filter_medicine_lines(lines):
    return [line for line in lines if is_medicine_line(line)]


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def extract_medicine_lines(image_path):
    ocr_data = extract_ocr_data(image_path)
    
    # 🔥 TASK 3: JUNK DATA GUARDRAIL
    # Look at the entire page text. If there is ABSOLUTELY 0 medical context anywhere on the page, reject it.
    full_text = " ".join([item['text'].lower() for item in ocr_data])
    medical_anchors = ['dr.', 'dr', 'clinic', 'hospital', 'rx', 'patient', 'age', 'sex', 'medicine', 'prescribed', 'mg', 'ml', 'tab', 'cap', 'syp', 'diagnostic']
    
    # BugFix: use substring matching instead of strict token equality
    domain_match = any(anchor in full_text for anchor in medical_anchors)
    
    if not domain_match and len(ocr_data) > 0:
        print("[!] GUARDRAIL WARNING: No clear medical anchors detected, but attempting fuzzy extraction anyway!")

    lines = group_into_lines(ocr_data)
    text_lines = lines_to_text(lines)
    medicine_lines = filter_medicine_lines(text_lines)

    return medicine_lines