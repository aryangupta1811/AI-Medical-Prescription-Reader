import pandas as pd
import json
import os

INPUT_PATH = "data/raw/Medicine_Details.csv"
OUTPUT_PATH = "data/processed/medicine_dictionary_clean.json"

# 🔹 CHECK FILE
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError("❌ Medicine_Details.csv not found in data/raw/")

df = pd.read_csv(INPUT_PATH)

print("Columns found:", df.columns.tolist())

# 🔹 CLEAN COLUMN NAMES
df.columns = [c.strip().lower() for c in df.columns]

clean_data = []

for _, row in df.iterrows():
    name = str(row.get("medicine name", "")).strip().lower()

    if not name or name == "nan":
        continue

    entry = {
        "name": name,
        "composition": str(row.get("composition", "")).strip(),
        "uses": str(row.get("uses", "")).strip(),
        "side_effects": str(row.get("side_effects", "")).strip(),
        "excellent_review": str(row.get("excellent review %", "")).strip(),
        "average_review": str(row.get("average review %", "")).strip(),
        "poor_review": str(row.get("poor review %", "")).strip(),
    }

    clean_data.append(entry)

# 🔹 REMOVE DUPLICATES
unique = {}
for item in clean_data:
    unique[item["name"]] = item

final_data = list(unique.values())

# 🔹 SAVE JSON
os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print("\n✅ Dictionary built successfully")
print(f"Total medicines: {len(final_data)}")