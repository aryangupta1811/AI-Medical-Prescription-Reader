import pandas as pd
import json
import re

INPUT_PATH = "data/raw/Medicine_Details.csv"
OUTPUT_PATH = "data/processed/final_knowledge_base.json"

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"\b\d+.*", "", text)
    text = re.sub(r"(tablet|tab|capsule|cap|syrup|injection|cream|gel)", "", text)
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(INPUT_PATH)
df.columns = [c.strip().lower() for c in df.columns]

knowledge = {}

for _, row in df.iterrows():
    composition = normalize(row.get("composition", ""))

    if not composition:
        continue

    # split multiple drugs
    parts = re.split(r"[,+/]", composition)

    for p in parts:
        generic = normalize(p)

        if len(generic) < 4:
            continue

        # keep first clean entry only
        if generic not in knowledge:
            knowledge[generic] = {
                "uses": str(row.get("uses", "")).strip(),
                "side_effects": str(row.get("side_effects", "")).strip()
            }

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(knowledge, f, indent=2)

print("✅ Final knowledge base created")
print("Total generics:", len(knowledge))