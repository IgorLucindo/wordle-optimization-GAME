import os
import shutil
import pandas as pd
from pathlib import Path

# Configuration
TO_ADD_DIR = Path("_to_add")
APPROVED_DIR = Path("instances")
REGISTRY_FILE = Path("instances_registry.csv")
TARGET_COLUMN = "target" # Update this to match your data's target column name

def check_resolvability(df: pd.DataFrame, target_col: str) -> bool:
    """
    Validates that no two different targets have the exact same feedbacks.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Group by all feedback features and count unique targets for each combination
    conflicts = df.groupby(feature_cols)[target_col].nunique()
    
    # If any feature combination points to more than 1 unique target, it's invalid
    if (conflicts > 1).any():
        return False
    return True

def process_new_instances():
    # Load or initialize the registry
    if REGISTRY_FILE.exists():
        registry = pd.read_csv(REGISTRY_FILE)
    else:
        registry = pd.DataFrame(columns=["instance_name", "num_targets", "num_features", "is_valid", "status"])

    new_records = []

    # Process each file in the _to_add folder
    for file_path in TO_ADD_DIR.glob("*.csv"):
        instance_name = file_path.stem
        
        # Skip if already in registry to avoid duplicates
        if instance_name in registry['instance_name'].values:
            print(f"Skipping {instance_name}: Already in registry.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            is_valid = check_resolvability(df, TARGET_COLUMN)
            
            # Record metadata
            new_records.append({
                "instance_name": instance_name,
                "num_targets": df[TARGET_COLUMN].nunique(),
                "num_features": len(df.columns) - 1,
                "is_valid": is_valid,
                "status": "Approved" if is_valid else "Rejected - Contradictory Feedbacks"
            })
            
            # Move file if valid
            if is_valid:
                shutil.move(str(file_path), str(APPROVED_DIR / file_path.name))
                print(f"✅ Validated and moved: {instance_name}")
            else:
                print(f"❌ Invalid (Contradictions found): {instance_name} - Left in _to_add/")
                
        except Exception as e:
            print(f"⚠️ Error processing {file_path.name}: {e}")

    # Append new records and save
    if new_records:
        updated_registry = pd.concat([registry, pd.DataFrame(new_records)], ignore_index=True)
        updated_registry.to_csv(REGISTRY_FILE, index=False)
        print("\nRegistry updated successfully.")
    else:
        print("\nNo new instances processed.")

if __name__ == "__main__":
    # Ensure directories exist
    TO_ADD_DIR.mkdir(exist_ok=True)
    APPROVED_DIR.mkdir(exist_ok=True)
    
    process_new_instances()