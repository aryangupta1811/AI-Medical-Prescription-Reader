from rapidfuzz import process, fuzz
import json

# 🔹 LOAD TOKEN INDEX
with open("data/processed/generic_tokens.json", "r") as f:
    token_map = json.load(f)

tokens = list(token_map.keys())


def match_medicine(text):
    text = text.lower()

    # 🔥 STEP 1: match against GENERIC TOKENS
    token, score, _ = process.extractOne(
        text,
        tokens,
        scorer=fuzz.ratio
    )

    if score < 60:
        return None, "LOW_CONFIDENCE"

    candidates = token_map[token]

    # 🔥 STEP 2: pick best candidate (shortest name bias)
    best = None
    best_score = -999

    for item in candidates:
        name = item["name"]

        s = fuzz.ratio(text, name)

        # bias towards simpler names
        score_final = s - len(name) * 0.5

        if score_final > best_score:
            best = item
            best_score = score_final

    return best, f"GENERIC_TOKEN_MATCH ({score:.1f})"


# 🔹 TEST
if __name__ == "__main__":
    samples = ["esonin", "esonix", "aceta", "alatrol"]

    print("\n🔍 FINAL MEDICAL MATCHING:\n")

    for s in samples:
        result, mode = match_medicine(s)

        if result:
            print(f"\n{s} → {result['name']}")
            print("Mode:", mode)
            print("Uses:", result["uses"])
        else:
            print(f"\n{s} → ❌ No reliable match")