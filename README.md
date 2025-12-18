# 🔠 Wordle Optimization

A Python project that optimizes solving the game **Wordle** by using a **polynomial-time strategy** to guess the hidden word in as few attempts as possible.

The solver applies optimization techniques to minimize the number of guesses while still ensuring correctness.

## ✨ Features

* Implements a **polytime approach** for Wordle optimization.
* Builds a **decision tree** for solving the game efficiently.
* Aims to solve in **the minimum number of guesses possible**.
* Includes a playable version of the game hosted online.

## 💾 Dataset

The word lists used for the possible Wordle answers and valid guesses are located in the **`dataset/` folder** of this project.

## ⚙️ Setup

1. **Install standard dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install CuPy (for GPU acceleration):**
   To enable GPU acceleration, you must install the `cupy` package that specifically matches your installed CUDA driver version.

   Please follow the instructions in the [Official CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).

## 🚀 Execution

* Play the game directly: [Wordle Optimization Game](https://igorlucindo.github.io/wordle-optimization-GAME)
* Build and evaluate decision tree:

```bash
python application/build_tree.py
```

* Evaluate our decision tree:

```bash
python application/eval_tree.py
```