import json
import re

INPUT_PATH = "data/processed/medicine_dictionary_clean.json"
OUTPUT_PATH = "data/processed/medicine_dictionary_normalized.json"


def clean_name(name):
    name = name.lower()

    # remove dosage (numbers + mg/ml/etc)
    name = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|g|iu)?\b", "", name)

    # remove forms
    remove_words = [
        "tablet", "tab", "capsule", "cap", "syrup",
        "injection", "cream", "gel", "ointment", "drops"
    ]

    for w in remove_words:
        name = name.replace(w, "")

    # remove extra symbols
    name = re.sub(r"[^a-z0-9 ]", "", name)

    # remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

normalized = []

for item in data:
    raw_name = item["name"]
    cleaned = clean_name(raw_name)

    if not cleaned:
        continue

    item["normalized_name"] = cleaned
    normalized.append(item)

# remove duplicates based on normalized_name
unique = {}
for item in normalized:
    unique[item["normalized_name"]] = item

final_data = list(unique.values())

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print("✅ Normalized dictionary created")
print("Total entries:", len(final_data))