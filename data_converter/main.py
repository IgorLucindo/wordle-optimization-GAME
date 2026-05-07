import pandas as pd
from pathlib import Path


# Configuration for the specific UCI datasets
CONFIG = {
    "car": {
        "file": "car+evaluation/car.data",
        "missing_val": None,
        "columns": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class_label"]
    },
    "house_votes_84": {
        "file": "congressional+voting+records/house-votes-84.data",
        "missing_val": "?",
        "columns": ["class_label", "handicapped-infants", "water-project", "budget-resolution",
                    "physician-fee", "el-salvador-aid", "religious-groups", "anti-satellite",
                    "nicaraguan-contras", "mx-missile", "immigration", "synfuels-cutback",
                    "education-spending", "superfund-right-to-sue", "crime", "duty-free", "south-africa"]
    },
    "soybean_small": {
        "file": "soybean+small/soybean-small.data",
        "missing_val": None,
        "columns": [f"attr_{i}" for i in range(1, 36)] + ["class_label"]
    }
}


def convert_uci_to_game_format(input_base_dir: str, output_base_dir: str):
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for name, conf in CONFIG.items():
        file_path = input_base / conf["file"]
        if not file_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue

        # 1. Read the raw data
        df = pd.read_csv(file_path, header=None, na_values=conf["missing_val"])
        df.columns = conf["columns"]

        # 2. Handle missing values
        df = df.fillna("unknown")

        feature_cols = [col for col in df.columns if col != "class_label"]

        # 3. ENCODE STRING CATEGORIES TO INTEGERS
        # This prevents the "invalid literal for int()" error in InstanceLoader
        for col in feature_cols:
            # pd.factorize assigns a unique integer to each unique string
            df[col] = pd.factorize(df[col])[0]

        # 4. Drop duplicate feature combinations 
        original_len = len(df)
        df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
        dropped_count = original_len - len(df)

        # 5. Create a unique 'target' identifier
        df['target'] = df['class_label'].astype(str) + "_" + df.index.astype(str)
        
        # 6. Drop the old class label and ensure 'target' is the first column
        df = df.drop(columns=["class_label"])
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index('target')))
        df = df[cols]

        # 7. Save to a dedicated folder
        dataset_dir = output_base / name
        dataset_dir.mkdir(exist_ok=True)
        
        output_file = dataset_dir / "attributes.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Converted {name}: Saved to {output_file}")
        if dropped_count > 0:
            print(f"   -> Dropped {dropped_count} duplicate feature combinations.")


if __name__ == "__main__":
    INPUT_DIR = "data_converter/raw_uci_data/"
    OUTPUT_DIR = "data/"
    
    convert_uci_to_game_format(INPUT_DIR, OUTPUT_DIR)