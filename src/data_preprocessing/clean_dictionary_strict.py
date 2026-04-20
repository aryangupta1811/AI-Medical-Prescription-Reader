import json
import re

INPUT_PATH = "data/processed/medicine_dictionary_clean.json"
OUTPUT_PATH = "data/processed/medicine_dictionary_strict.json"

def normalize(text):
    text = text.lower()
    text = re.sub(r"\b\d+.*", "", text)
    text = re.sub(r"(tablet|tab|capsule|cap|syrup|injection|cream|gel)", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_clean_generic(name):
    # reject multi-drug combinations
    if "+" in name or "," in name:
        return False

    # reject too long names (likely brand variants)
    if len(name.split()) > 2:
        return False

    return True

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = {}

for item in data:
    name = normalize(item["name"])

    if not name:
        continue

    if not is_clean_generic(name):
        continue

    # keep first occurrence only
    if name not in cleaned:
        cleaned[name] = item

final_data = list(cleaned.values())

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print("✅ Strict dictionary created")
print("Total clean entries:", len(final_data))
