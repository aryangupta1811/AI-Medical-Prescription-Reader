import json
import re

DICT_PATH = "data/processed/medicine_dictionary_clean.json"
OUTPUT_PATH = "data/processed/generic_expanded.json"

def extract_generics(composition):
    composition = composition.lower()

    # split multiple drugs
    parts = re.split(r"[,+/]", composition)

    generics = []
    for p in parts:
        p = re.sub(r"\b\d+.*", "", p)  # remove dosage
        p = re.sub(r"[^a-z ]", "", p).strip()

        if len(p) > 3:
            generics.append(p)

    return generics


with open(DICT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

generic_map = {}

for item in data:
    name = item["name"]
    composition = item["composition"]

    gens = extract_generics(composition)

    for g in gens:
        if g not in generic_map:
            generic_map[g] = set()

        generic_map[g].add(name)

# convert to list
for k in generic_map:
    generic_map[k] = list(generic_map[k])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(generic_map, f, indent=2)

print("✅ Expanded generic map created")
print("Total generics:", len(generic_map))