import json
import re

INPUT_PATH = "data/processed/medicine_dictionary_clean.json"
OUTPUT_PATH = "data/processed/generic_tokens.json"

def clean_token(text):
    text = text.lower()
    text = re.sub(r"\b\d+.*", "", text)  # remove dosage
    text = re.sub(r"[^a-z ]", "", text)
    text = text.strip()
    return text

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

token_map = {}

for item in data:
    composition = item["composition"]

    parts = re.split(r"[,+/]", composition)

    for p in parts:
        token = clean_token(p)

        if len(token) < 4:
            continue

        token_map.setdefault(token, []).append(item)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(token_map, f, indent=2)

print("✅ Generic token index created")
print("Total tokens:", len(token_map))