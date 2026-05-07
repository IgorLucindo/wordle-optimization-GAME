# 🛠️ UCI Data Converter

This directory contains the preprocessing pipeline to import and format classification datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for use with our optimal decision tree approximation.

By converting these tabular datasets into our generalized **Feedback Matrix** format (`feedbacks.csv`), we can use the same optimization engine built for Wordle to solve a wide variety of Active Sequential Testing and identification problems.

## 📖 Terminology: From Features to Feedbacks

Our solver does not look at static "features" in the traditional machine learning sense. Instead, it evaluates the results of **probes** applied to hidden **targets** using a discrete **feedback function**. To bridge the gap between UCI datasets and our game solver, this converter maps the data as follows:

* **Targets (Rows):** The hidden entities we are trying to identify (e.g., a specific instance of a car or a specific voting record).
* **Probes (Columns):** The questions we are allowed to ask or the tests we can perform (e.g., "How many doors?" or "How did they vote on the budget?").
* **Feedbacks (Cells):** The discrete integer result returned when applying that Probe to that Target.

## 🧩 What Makes a Valid Instance?

Our framework is designed to find a path to a **single, uniquely identifiable target** at every terminal leaf. Because of this, not all standard machine learning datasets work out-of-the-box. 

For a problem instance to be valid for our solver, it must satisfy the rule of **Resolvability**:
> **No two distinct targets can share the exact same feedback vector.**

If `Target A` and `Target B` return identical values for every single available probe, the decision tree will never be able to fully separate them, creating an unsolvable contradiction. 

To ensure valid instances, this converter automatically:
1. Translates string categories into integer feedback codes.
2. Scans for duplicate feedback vectors across the dataset.
3. Drops redundant rows to guarantee that every target in the final `feedbacks.csv` is uniquely identifiable.

## 📦 Supported Datasets

The `main.py` script is currently configured to process the following UCI datasets:

* **Car Evaluation** (`car`)
* **Congressional Voting Records 1984** (`house_votes_84`)
* **Soybean Small** (`soybean_small`)

## ⚙️ Setup & Execution

### 1) Prepare the Raw Data
Ensure the raw UCI dataset folders are placed inside the `data_converter/raw_uci_data/` directory. For example:
```text
data_converter/raw_uci_data/
├── car+evaluation/
│   └── car.data
├── congressional+voting+records/
│   └── house-votes-84.data
└── soybean+small/
    └── soybean-small.data
```

### 2) Run the Converter
Execute the Python script from the root of the repository:
```bash
python data_converter/main.py
```

### 3) Output
The script will process the data, handle missing values (replacing them with an `"unknown"` category to be converted to an integer code), drop unresolvable duplicate rows, and output the clean instances to the main `data/` directory.

You will see output similar to this:
```plaintext
✅ Converted car: Saved to data/car/feedbacks.csv
   -> Dropped X duplicate feedback vectors.
```
Once converted, you can optimize these datasets using the main solver:
```bash
python application/build_tree.py --data car --k 10
```