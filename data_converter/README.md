# UCI Data Converter

This directory contains the preprocessing pipeline to import and format classification datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for use with the solver.

## How It Fits Into the Architecture

The solver is built around a **Feedback Matrix** `F[target, guess]`: a 2D table of integers where each cell holds the discrete response returned when guess `j` is applied to target `i`. For rule-based games like Wordle or Mastermind, this matrix is computed on-the-fly from compact rules (ternary comparison, peg counting). For tabular datasets, the data *is* the feedback matrix — no computation needed.

This converter bridges the two worlds. It takes a raw UCI dataset and produces `attributes.csv`, which the solver loads directly as its feedback matrix. The columns are guesses (attributes/questions), the rows are targets (individual instances), and the cells are the integer responses.

## Terminology

| UCI / ML term | Solver term |
|---|---|
| Dataset row / instance | **Target** — the hidden entity being identified |
| Feature / column | **Guess** — a question that can be asked about any target |
| Feature value | **Feedback** — the discrete integer response |

## What Makes a Valid Instance

The solver must be able to reach a **unique** target at every leaf. This requires **Resolvability**:

> No two distinct targets may share the exact same feedback vector.

If targets A and B return identical responses for every available guess, no sequence of questions can distinguish them. The converter enforces this automatically by dropping duplicate rows before writing output.

## Supported Datasets

| Name | Source | Targets | Guesses |
|---|---|---|---|
| `car` | Car Evaluation | car configurations | buying price, safety, etc. |
| `house_votes_84` | Congressional Voting Records 1984 | voting records | 16 bill votes |
| `soybean_small` | Soybean (Small) | disease instances | 35 symptom attributes |

## Adding a New Dataset

1. Add an entry to `data_converter/config.json` with the file path, `label_col` (index of the class label column), and `guesses` (list of column names, or `null` to auto-generate `attr_0`, `attr_1`, ...).
2. Add a `rules.json` to `data/<name>/` specifying `"format": "csv"` and `"feedback_engine": "attribute_matrix"`.
3. Run the converter. The output `attributes.csv` is immediately usable by the solver.

No changes to the solver itself are needed.

## Setup & Usage

### 1. Prepare raw data

Place the raw UCI dataset folders inside `data_converter/raw_uci_data/`:

```
data_converter/raw_uci_data/
├── car+evaluation/
│   └── car.data
├── congressional+voting+records/
│   └── house-votes-84.data
└── soybean+small/
    └── soybean-small.data
```

### 2. Run the converter

From the repository root:

```bash
python data_converter/main.py
```

### 3. Output

Each dataset is written to `data/<name>/attributes.csv`. Missing values are assigned an `"unknown"` category; duplicate rows are dropped. Example output:

```
Converted car: Saved to data/car/attributes.csv
Converted house_votes_84: Saved to data/house_votes_84/attributes.csv
  -> Dropped 3 duplicate feature combinations.
Converted soybean_small: Saved to data/soybean_small/attributes.csv
```

Once converted, run the solver as usual:

```bash
python application/build_tree.py --data car --k 10
```