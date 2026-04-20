print("[START] SCORING PIPELINE ACTIVE")

import os
import json

from src.pipeline.line_processing import extract_medicine_lines
from src.postprocessing.final_pipeline import final_match


# ----------------------------------------
# CONFIG
# ----------------------------------------
IMAGE_PATH = "test_prescription.jpg"
SCORE_THRESHOLD = 1.15   # Accurately tuned for OCR Twin-Engine normalization


# ----------------------------------------
# SCORING FUNCTION
# ----------------------------------------
from rapidfuzz import fuzz

def compute_match_score(input_line, match):
    if not match:
        return 0.0

    name = match.get("generic", "").lower()
    fuzzy_conf = match.get("confidence", 0) / 100.0  # normalize

    import re
    input_tokens = set(input_line.lower().split())
    name_tokens = set(name.replace('-', ' ').replace('/', ' ').split())
    
    # ------------------------
    # CRITICAL FIX: Brand Name Vectorization!
    # Because OCR captures Brand Names ("Ace"), computing string distance literally fails against purely Generic names ("Paracetamol")
    # We must explicitly add the captured dictionary trigger string to the expected name_tokens matrix natively!
    matched_token = match.get('matched_token', '').lower()
    if matched_token:
        name_tokens.update(matched_token.replace('-', ' ').replace('/', ' ').split())

    # ------------------------
    # 1. Fuzzy Token overlap score
    # ------------------------
    if len(input_tokens) == 0:
        token_score = 0
    else:
        # Filter out generic prescription stop words dynamically
        stop_words = {"dr", "dr.", "patient", "name", "date", "signed", "by", "rx", "age", "clinic", "hospital", "medical", 
                      "the", "and", "for", "tab", "cap", "syr", "inj", "day", "b.i.d", "t.i.d", "mg", "ml"}
        clean_input = [w for w in input_tokens if w not in stop_words and len(w) > 2]

        if not clean_input or len(name_tokens) == 0:
            token_score = 0
        else:
            matched_tokens_count = 0
            for i_tok in clean_input:
                best_tok_score = 0
                for n_tok in name_tokens:
                    s1 = fuzz.ratio(i_tok, n_tok)
                    # Exclusively empower mathematical partial ratios strictly if the token is robust enough (4+ chars)
                    # AND it mathematically aligns with the front-half architecture of the word (preventing internal substring attractors)
                    s2 = fuzz.partial_ratio(i_tok, n_tok) if len(i_tok) > 3 and (n_tok.startswith(i_tok[:2]) or i_tok.startswith(n_tok[:2])) else 0
                    
                    # Phase 3 NLP Scale: Physical Soundex Match Matrix (Overrides Visual OCR Spelling Failures!)
                    import jellyfish
                    s3 = 100 if len(i_tok) > 3 and jellyfish.soundex(i_tok) == jellyfish.soundex(n_tok) else 0
                    
                    s = max(s1, s2, s3)
                    
                    if s > best_tok_score:
                        best_tok_score = s
                if best_tok_score > 75:  # fuzzy threshold
                    matched_tokens_count += 1
                
        # Normalize mathematically against the pure dictionary length, NOT the garbage-heavy input length!
        token_score = min(1.0, matched_tokens_count / len(name_tokens))

    # ------------------------
    # 2. Numeric match score (Dosage checks)
    # ------------------------
    input_numbers = [w for w in input_tokens if re.search(r'\d', w)]
    name_numbers = [w for w in name_tokens if re.search(r'\d', w)]

    if input_numbers and name_numbers:
        matches = sum(1 for n in input_numbers if n in name_numbers)
        numeric_score = matches / len(input_numbers)
    elif input_numbers and not name_numbers:
        numeric_score = 0.0 # Bypassed - Doctors write dosages natively but generics lack them!
    else:
        numeric_score = 0.0

    # ------------------------
    # 3. Final score
    # ------------------------
    final_score = token_score + numeric_score + fuzzy_conf

    return final_score


# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------
def run_pipeline(image_path):
    print("\n[INFO] Starting Prescription Processing...\n")

    try:
        with open("data/processed/final_knowledge_base.json", "r") as f:
            kb = json.load(f)
    except Exception:
        kb = {}

    medicine_lines = extract_medicine_lines(image_path)

    print("[DEBUG] Extracted Medicine Lines:")
    for i, line in enumerate(medicine_lines):
        print(f"  {i+1}. {line}")

    if not medicine_lines:
        print("\n[WARNING] No medicine lines detected.\n")
        return []

    results = []

    for line in medicine_lines:
        match = final_match(line)

        score = compute_match_score(line, match)

        print(f"\n[DEBUG] Line: {line}")
        print(f"        Candidate: {match}")
        print(f"        Score: {score:.2f}")

        if score >= SCORE_THRESHOLD:
            # Embed Knowledge Base parameters
            generic_name = match.get('generic', '').lower()
            info = kb.get(generic_name, {})
            match['uses'] = info.get('uses', 'Clinical specifics currently unavailable.')
            match['side_effects'] = info.get('side_effects', 'Safety data temporarily unavailable.')

            results.append({
                "input_line": line,
                "matched_medicine": match,
                "score": score
            })
        else:
            results.append({
                "input_line": line,
                "matched_medicine": None,
                "score": score
            })

    return results


# ----------------------------------------
# DISPLAY
# ----------------------------------------
def display_results(results):
    print("\n================ FINAL OUTPUT ================\n")

    if not results:
        print("No medicines detected.\n")
        return

    # Load mapping database
    try:
        with open("data/processed/final_knowledge_base.json", "r") as f:
            kb = json.load(f)
    except Exception:
        kb = {}

    for i, item in enumerate(results):
        print(f"[{i+1}] Input Line: {item['input_line']}")
        print(f"     -> Score: {item['score']:.2f}")

        match = item["matched_medicine"]
        if match:
            generic = match.get('generic', '').lower()
            print(f"     -> Matched: {generic.title()}")
            
            # Fetch rich details
            info = kb.get(generic, {})
            if info:
                print(f"     -> Uses: {info.get('uses', 'No data')}")
                print(f"     -> Side Effects: {info.get('side_effects', 'No data')}")
        else:
            print("     -> Matched: None (Below threshold)")

        print()


# ----------------------------------------
# ENTRY POINT
# ----------------------------------------
if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        exit()

    results = run_pipeline(IMAGE_PATH)

    display_results(results)

    with open("output_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n[INFO] Results saved to output_results.json\n")