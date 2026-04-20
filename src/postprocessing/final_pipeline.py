from rapidfuzz import fuzz
import json

# =========================
# LOAD DATA
# =========================
with open("data/processed/generic_to_brand.json", "r") as f:
    raw_data = json.load(f)

# =========================
# NORMALIZE TO VOCABULARY DICT
# =========================
med_dict = {}

for generic, brands in raw_data.items():
    med_dict[generic.lower()] = generic.lower()
    for brand in brands:
        if brand.strip():
            med_dict[brand.lower().strip()] = generic.lower()

medicine_list = list(med_dict.keys())

# =========================
# FINAL MATCH FUNCTION
# =========================
def final_match(predictions):
    if isinstance(predictions, str):
        predictions = [predictions]

    best_match = None
    best_score = 0

    for pred in predictions:

        pred = pred.lower().strip()

        for med in medicine_list:

            # Prevent short 3-letter brands from mathematically sliding across long sentences and generating fake matches
            pr = fuzz.partial_ratio(pred, med) if len(med) > 3 else fuzz.ratio(pred, med)
            tsr = fuzz.token_set_ratio(pred, med)
            score = max(pr, tsr)

            if score > best_score:
                best_score = score
                best_match = med

    # threshold (balanced)
    if best_score < 50:
        return None

    return {
        "generic": med_dict.get(best_match, best_match),
        "matched_token": best_match,
        "confidence": best_score
    }