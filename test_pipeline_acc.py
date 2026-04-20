import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.pipeline.line_processing import extract_ocr_data
from src.postprocessing.final_pipeline import final_match
from src.pipeline.hybrid_pipeline import compute_match_score

def evaluate():
    print("Loading test dataset...")
    df = pd.read_csv("data/raw/test_labels.csv")
    correct = 0
    total = 0
    
    # We will test on a robust sample limit so it mathematically runs efficiently
    sample_limit = 200
    df = df.sample(n=sample_limit, random_state=42)
    
    for idx, row in df.iterrows():
        img_file = row['IMAGE']
        true_generic = str(row['GENERIC_NAME']).lower().strip()
        
        img_path = os.path.join("data/raw/test_images", img_file)
        if not os.path.exists(img_path):
            continue
            
        total += 1
        
        # 1. Pipeline Extraction (EasyOCR + PyTorch Native)
        try:
            ocr_blocks = extract_ocr_data(img_path)
            if not ocr_blocks:
                continue
                
            full_text = " ".join([b['text'] for b in ocr_blocks]).lower()
            
            # 2. Pipeline Fuzzy Matching Matrix
            match = final_match(full_text)
            if not match:
                continue
                
            pred_generic = str(match['generic']).lower().strip()
            
            # 3. Hybrid Verification Guardrail
            score = compute_match_score(full_text, match)
            
            # Only count as correct if it survived ALL mathematical guardrails!
            if pred_generic == true_generic and score >= 1.05:
                correct += 1
            else:
                if total <= 10:  # Just print the first few failures to analyze why they died
                    print(f"\n[FAILURE ANALYSIS] Image: {img_file}")
                    print(f"  True Label (Generic): {true_generic}")
                    print(f"  OCR Extraction: '{full_text}'")
                    print(f"  Matched Target: '{pred_generic}' (Score: {score:.2f})")
                
        except Exception as e:
            continue

    if total == 0:
        print("Dataset images not found.")
        return

    accuracy = (correct / total) * 100
    print("\n==================================")
    print(" END-TO-END PIPELINE ACCURACY TEST")
    print("==================================")
    print(f"Total Prescriptions OCR'd: {total}")
    print(f"Correct Generic Matches: {correct}")
    print(f"FINAL TRUE PIPELINE ACCURACY: {accuracy:.2f}%")
    print("==================================")

if __name__ == "__main__":
    evaluate()
