import os
import pandas as pd
from pathlib import Path

def debug_pipeline():
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    labels_out = processed_dir / "labels.txt"
    
    print("\n" + "="*50)
    print("--- 1. VERIFY AND FIX DIRECTORY NAMES ---")
    print("="*50)
    
    renames = [
        ("training_words", "train_images"),
        ("validation_words", "val_images"),
        ("testing_words", "test_images")
    ]
    for old, new in renames:
        old_path = raw_dir / old
        new_path = raw_dir / new
        if old_path.exists():
            if not new_path.exists():
                old_path.rename(new_path)
                print(f"[RENAME SUCCESS] {old} -> {new}")
            else:
                print(f"[SKIP] Both {old} and {new} exist. Skipping rename.")
        else:
            if new_path.exists():
                print(f"[OK] {new} already exists. No rename needed.")
            else:
                print(f"[MISSING] Neither {old} nor {new} exist in {raw_dir}.")

    print("\n" + "="*50)
    print("--- 2. DEBUG CSV LOADING ---")
    print("="*50)
    
    train_csv = raw_dir / "train_labels.csv"
    val_csv = raw_dir / "val_labels.csv"
    
    df_list = []
    for csv_path in [train_csv, val_csv]:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                print(f"✅ Loaded {len(df)} rows from {csv_path.name}")
                print(f"First 5 rows of {csv_path.name}:")
                print(df.head())
                
                # Check required columns
                missing_cols = [c for c in ["IMAGE", "MEDICINE_NAME", "GENERIC_NAME"] if c not in df.columns]
                if missing_cols:
                    print(f"❌ [WARNING] Missing columns in {csv_path.name}: {missing_cols}")
                    
                df_list.append(df)
            except Exception as e:
                print(f"❌ [ERROR] Could not read {csv_path.name}: {e}")
        else:
            print(f"❌ [WARNING] Missing CSV file: {csv_path.name}")
            
    if not df_list:
        raise ValueError("No CSVs loaded! Terminating immediately.")

    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("\n" + "="*50)
    print("--- 3. VERIFY IMAGE PATH MATCHING ---")
    print("="*50)
    
    search_dirs = [raw_dir / "train_images", raw_dir / "val_images"]
    
    found_count = 0
    missing_count = 0
    valid_records = []
    
    for _, row in combined_df.iterrows():
        if 'IMAGE' not in row or 'MEDICINE_NAME' not in row:
            missing_count += 1
            continue
            
        img_name = str(row['IMAGE']).strip()
        medicine_name = row['MEDICINE_NAME']
        
        # Check if image actually exists in train or val folders
        found = False
        for sdir in search_dirs:
            if (sdir / img_name).exists():
                found = True
                break
                
        if found:
            found_count += 1
            valid_records.append((img_name, medicine_name))
        else:
            missing_count += 1
            
    print(f"Images FOUND:   {found_count}")
    print(f"Images MISSING: {missing_count}")
    
    print("\n" + "="*50)
    print("--- 4. TEMPORARY BYPASS CLEANING ---")
    print("="*50)
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(labels_out, "w", encoding="utf-8") as f:
        for img_name, med_name in valid_records:
            if pd.isna(med_name):
                label = ""
            else:
                label = str(med_name).lower().strip()
                
            f.write(f"{img_name} {label}\n")
            
    print(f"Writing complete. Bypassed strict regex cleaning to write directly to {labels_out.name}")
    
    print("\n" + "="*50)
    print("--- 5. PRINT FINAL COUNTS ---")
    print("="*50)
    
    print(f"Total labels written: {len(valid_records)}")
    if len(valid_records) > 0:
        print(f"Sample 10 lines from {labels_out.name}:")
        with open(labels_out, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[:10]:
                print(line.strip())
                
    print("\n" + "="*50)
    print("--- 6. FAIL LOUDLY ---")
    print("="*50)
    
    if found_count == 0:
        raise RuntimeError("🚨 FATAL ERROR: No images found! Pipeline failed.")
    
    if len(valid_records) == 0:
        raise RuntimeError("🚨 FATAL ERROR: No labels written! Pipeline failed.")
        
    print("✅ Debug script completed successfully. Labels exist!")


if __name__ == "__main__":
    debug_pipeline()
