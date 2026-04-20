import os
from pathlib import Path

def create_structure():
    # Base directory is the directory containing this script (project root)
    base_dir = Path(__file__).parent.resolve()
    
    # Define directories to create
    directories = [
        "data/raw/train_images",
        "data/raw/val_images",
        "data/raw/test_images",
        "data/processed/images",
        "src/data_preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "models/checkpoints",
        "models/final",
        "outputs/predictions",
        "outputs/logs",
        "outputs/metrics",
        "notebooks",
        "configs"
    ]
    
    # Define files to create (like CSV, JSON, python files in the spec)
    files = [
        "data/raw/train_labels.csv",
        "data/raw/val_labels.csv",
        "data/raw/test_labels.csv",
        "data/raw/medicine_dictionary.csv",
        "data/processed/labels.txt",
        "data/processed/medicine_dictionary_clean.json",
        "data/processed/generic_to_brand.json",
        "main.py"
    ]
    
    print("Initializing project structure...")
    print("-" * 50)
    
    # 1. Create Directories & .gitkeep
    for dir_path in directories:
        full_dir_path = base_dir / dir_path
        
        if not full_dir_path.exists():
            full_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            # Create .gitkeep in the empty directory
            gitkeep_path = full_dir_path / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
                print(f"Created file: {dir_path}/.gitkeep")
        else:
            print(f"Directory already exists: {dir_path}")
            
    # 2. Create Placeholder Files
    for file_path in files:
        full_file_path = base_dir / file_path
        
        # Ensure parent directory exists for file (safeguard)
        parent_dir = full_file_path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            
        if not full_file_path.exists():
            full_file_path.touch()
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

    print("-" * 50)
    print("Project structure generation complete!")

if __name__ == "__main__":
    create_structure()
