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

## 🚀 Execution

* Play the game directly: [Wordle Optimization Game](https://igorlucindo.github.io/wordle-optimization-GAME)
* Create and evaluate decision tree (NEED TO BE UPDATED):

```bash
python application/main.py
```

## 📝 How it works

* Uses **entropy-based scoring** to pick guesses that maximize expected information gain.
* Breaks ties using **minimax partition size** and lexicographical order.
* Runs in **polynomial time per guess**, making it fast enough to simulate thousands of games.