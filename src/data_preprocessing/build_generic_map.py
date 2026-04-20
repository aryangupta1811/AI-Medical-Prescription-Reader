import pandas as pd
import json
import os

TRAIN_PATH = "data/raw/train_labels.csv"
VAL_PATH = "data/raw/val_labels.csv"
OUTPUT_PATH = "data/processed/generic_to_brand.json"

# 🔹 LOAD DATA
dfs = []

for path in [TRAIN_PATH, VAL_PATH]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
    else:
        print(f"⚠️ Missing file: {path}")

if len(dfs) == 0:
    raise RuntimeError("❌ No label CSVs found")

df = pd.concat(dfs, ignore_index=True)

# 🔹 CLEAN COLUMN NAMES
df.columns = [c.strip().lower() for c in df.columns]

# 🔹 CHECK REQUIRED
if "medicine_name" not in df.columns or "generic_name" not in df.columns:
    raise RuntimeError("❌ Required columns missing: medicine_name, generic_name")

generic_map = {}

for _, row in df.iterrows():
    brand = str(row["medicine_name"]).strip().lower()
    generic = str(row["generic_name"]).strip().lower()

    if not brand or not generic or brand == "nan" or generic == "nan":
        continue

    if generic not in generic_map:
        generic_map[generic] = set()

    generic_map[generic].add(brand)

# 🔹 CONVERT SET → LIST
for k in generic_map:
    generic_map[k] = list(generic_map[k])

# 🔹 SAVE
os.makedirs("data/processed", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(generic_map, f, indent=2)

print("\n✅ Generic → Brand map created")
print(f"Total generics: {len(generic_map)}")