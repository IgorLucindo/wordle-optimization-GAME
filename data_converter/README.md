# UCI Data Converter

This directory contains the preprocessing pipeline to import and format classification datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) for use with the solver.

## Why Use UCI Datasets? (Near-Optimal Policies)

Standard Machine Learning decision trees aim to predict a general label and actively avoid overfitting. This solver does the exact opposite: it treats every single dataset row as a unique target and rapidly finds a close-to-optimal, highly minimized path to identify *that exact row*.

By applying this solver to tabular data, it acts as an engine for building **Near-Optimal Policies** and **Diagnostic Keys**.

The best datasets for this solver share two characteristics:
1. **Discrete/Categorical Features:** The solver relies on a discrete Feedback Matrix (e.g., ternary responses or exact matches).
2. **High-Cost Questions:** Scenarios where minimizing the number of questions asked is highly valuable in the real world (e.g., medical tests, time-consuming mechanical checks, or user friction).

**Ideal Use Cases & Recommended Datasets:**
* **Biological/Taxonomic Keys** (`zoo`, `soybean_large`): Generates a highly efficient sequence of observations a biologist should make to identify a species or plant disease in the field.
* **Medical Diagnosis** (`breast-cancer`, `dermatology`): Finds a streamlined flowchart that requires a near-minimum number of clinical tests to reach a specific diagnosis.
* **Fault Isolation** (`car`): Acts as a troubleshooting tree, minimizing the number of sensor checks needed to identify a mechanical state.
* **Preference Routing/20 Questions** (`mushroom`, `lenses`): Interactive recommendation engines that narrow down a massive space of possibilities with very few user prompts.

## How It Fits Into the Architecture

The solver is built around a **Feedback Matrix** `F[target, guess]`: a 2D table of integers where each cell holds the discrete response returned when guess `j` is applied to target `i`. For rule-based games like Wordle or Mastermind, this matrix is computed on-the-fly from compact rules (ternary comparison, peg counting). For tabular datasets, the data *is* the feedback matrix — no computation needed.

This converter bridges the two worlds. It takes a raw UCI dataset and automatically produces both `attributes.csv` (the feedback matrix) and `rules.json` (the solver configuration). The columns are guesses (attributes/questions), the rows are targets (individual instances), and the cells are the integer responses.

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

## Adding a New Dataset

Adding a new dataset is highly automated. The converter handles missing values (mapping them to an `"unknown"` category), encodes string categories to integers, drops unresolvable duplicates, and auto-generates the solver's configuration file.

**Example: Adding the `mushroom` dataset**

1. **Download:** Place the raw data file inside `data_converter/raw_uci_data/mushroom/agaricus-lepiota.data`.
2. **Configure:** Add an entry to `data_converter/config.json` specifying the file path, the index of the class label column, and the character used for missing values:
   ```json
   "mushroom": {
       "file": "mushroom/agaricus-lepiota.data",
       "label_col": 0,
       "missing_val": "?"
   }
   ```
   *(Note: You can also pass a `"guesses"` list with exact column names, otherwise it auto-generates `attr_0`, `attr_1`, etc.)*
3. **Run the Converter:** The script automatically generates `attributes.csv` and `rules.json` inside `data/mushroom/`. It is immediately usable by the solver!

## Setup & Usage

### 1. Prepare raw data

Place the raw UCI dataset folders inside `data_converter/raw_uci_data/`:

```text
data_converter/raw_uci_data/
└── mushroom/
    └── agaricus-lepiota.data
```

### 2. Run the converter

From the repository root:

```bash
python data_converter/main.py
```

### 3. Output

Each dataset is written to `data/<name>/`. Missing values are assigned an `"unknown"` category; duplicate rows are dropped. Example output:

```text
Converted mushroom: Saved to data\mushroom (attributes.csv & rules.json)
```

Once converted, run the solver as usual:

```bash
python application/build_tree.py --data mushroom --k 10
```