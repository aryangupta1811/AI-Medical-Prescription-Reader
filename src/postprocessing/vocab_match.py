from rapidfuzz import process, fuzz
import json
import pandas as pd

# 🔹 LOAD LABEL VOCAB (CRITICAL)
df = pd.read_csv("data/raw/train_labels.csv")
df.columns = [c.lower() for c in df.columns]

vocab = list(set(df["medicine_name"].str.lower().tolist()))

# 🔹 LOAD GENERIC MAP
with open("data/processed/generic_to_brand.json", "r") as f:
    generic_map = json.load(f)

# 🔹 REVERSE MAP
brand_to_generic = {}
for g, brands in generic_map.items():
    for b in brands:
        brand_to_generic[b] = g

# 🔹 LOAD DICTIONARY
with open("data/processed/medicine_dictionary_clean.json", "r") as f:
    med_data = json.load(f)

dict_names = [m["name"].lower() for m in med_data]


def match_medicine(text):
    text = text.lower()

    # 🔥 STEP 1: match within VOCAB ONLY
    brand, score, _ = process.extractOne(
        text,
        vocab,
        scorer=fuzz.ratio
    )

    if score < 60:
        return None, "NO_VOCAB_MATCH"

    # 🔥 STEP 2: map to generic
    generic = brand_to_generic.get(brand)

    if not generic:
        return None, "NO_GENERIC"

    # 🔥 STEP 3: map to dictionary
    match, d_score, idx = process.extractOne(
        generic,
        dict_names,
        scorer=fuzz.ratio
    )

    if d_score < 70:
        return None, "NO_DICT_MATCH"

    return med_data[idx], f"VOCAB_MATCH ({score:.1f})"


# 🔹 TEST
if __name__ == "__main__":
    samples = ["esonin", "esonix", "aceta", "alatrol"]

    print("\n🔍 CONTROLLED MATCHING:\n")

    for s in samples:
        result, mode = match_medicine(s)

        if result:
            print(f"\n{s} → {result['name']}")
            print("Mode:", mode)
            print("Uses:", result["uses"])
        else:
            print(f"\n{s} → ❌ No reliable match")