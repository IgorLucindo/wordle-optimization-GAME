import json
import pandas as pd
from pathlib import Path


def convert_uci_to_game_format(config_path: str, input_base_dir: str, output_base_dir: str):
    with open(config_path) as f:
        config = json.load(f)

    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for name, conf in config.items():
        file_path = input_base / conf["file"]
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        # 1. Read the raw data
        separator = conf.get("sep", ",")
        df = pd.read_csv(file_path, header=None, na_values=conf.get("missing_val"), sep=separator, engine="python")

        # 2. Split label column from features
        n_cols = len(df.columns)
        label_idx = conf["label_col"] % n_cols
        feature_idxs = [i for i in range(n_cols) if i != label_idx]

        labels = df.iloc[:, label_idx]
        features = df.iloc[:, feature_idxs].copy()

        # 3. Name feature columns
        feature_names = conf.get("guesses")
        if feature_names:
            features.columns = feature_names
        else:
            features.columns = [f"attr_{i}" for i in range(len(feature_idxs))]

        # 4. Handle missing values
        features = features.fillna("unknown")

        # 5. Encode string categories to integers
        for col in features.columns:
            features[col] = pd.factorize(features[col])[0]

        # 6. Drop duplicate feature combinations
        original_len = len(features)
        keep = ~features.duplicated()
        features = features[keep].reset_index(drop=True)
        labels = labels[keep].reset_index(drop=True)
        dropped_count = original_len - len(features)

        # 7. Build target identifier and prepend as first column
        features.insert(0, "target", labels.astype(str) + "_" + features.index.astype(str))

        # 8. Save
        dataset_dir = output_base / name
        dataset_dir.mkdir(exist_ok=True)
        output_file = dataset_dir / "attributes.csv"
        features.to_csv(output_file, index=False)

        rules = {
            "format": "csv",
            "feedback_engine": "attribute_matrix",
            "guesses_include_targets": False,
            "constrained_guessing": False,
            "data_files": {
                "dataset": "attributes.csv",
                "has_header": True
            },
            "description": f"Auto-converted UCI {name.replace('_', ' ').title()} dataset"
        }
        
        rules_file = dataset_dir / "rules.json"
        with open(rules_file, "w") as f:
            json.dump(rules, f, indent=2)

        print(f"Converted {name}: Saved to {dataset_dir} (attributes.csv & rules.json)")
        if dropped_count > 0:
            print(f"  -> Dropped {dropped_count} duplicate feature combinations.")


if __name__ == "__main__":
    CONFIG_FILE = "data_converter/config.json"
    INPUT_DIR = "data_converter/raw_uci_data/"
    OUTPUT_DIR = "data/"

    convert_uci_to_game_format(CONFIG_FILE, INPUT_DIR, OUTPUT_DIR)
