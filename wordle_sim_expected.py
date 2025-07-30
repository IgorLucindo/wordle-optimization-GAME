from collections import Counter, defaultdict
from typing import List, Tuple
import math
import time

def load_words_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf‑8") as f:
        return [line.strip().lower() for line in f]

def feedback(guess: str, target: str) -> str:
    """Return Wordle feedback string of 'G', 'Y', 'B'."""
    result = ["B"] * len(target)
    t_count = Counter(target)

    # greens
    for i, g in enumerate(guess):
        if g == target[i]:
            result[i] = "G"
            t_count[g] -= 1

    # yellows
    for i, g in enumerate(guess):
        if result[i] == "B" and t_count[g] > 0:
            result[i] = "Y"
            t_count[g] -= 1

    return "".join(result)


def filter_candidates(cands: List[str], guess: str, fb: str) -> List[str]:
    return [w for w in cands if feedback(guess, w) == fb]

def expected_remaining_candidates_len(guess: str, candidates: List[str]) -> float:
    """Compute E[ |L'| ] for a single guess given current candidate list."""
    pattern_counts = Counter(feedback(guess, l) for l in candidates)
    n = len(candidates)
    return sum(c * c for c in pattern_counts.values()) / n

def best_guess_by_expected_candidates(
    candidates: List[str],                    # L
    all_words: List[str]        # W (None ⇒ use L)
) -> Tuple[str, float]:
    if all_words is None:
        all_words = candidates

    best_exp = float("inf")
    best_w   = None
    for w in all_words:
        exp_size = expected_remaining_candidates_len(w, candidates)
        if exp_size < best_exp:
            best_exp, best_w = exp_size, w
        # tie‑break: prefer a word that is itself a candidate
        elif exp_size == best_exp and best_w not in candidates and w in candidates:
            best_w = w
    return best_w, best_exp

def simulate_game(target: str,
                  word_list: List[str],       # candidate solutions (L₀)
                  guess_pool: List[str]):     # allowed guesses (W) given the feedback constraints
    if guess_pool is None:
        guess_pool = word_list

    candidates = word_list.copy()
    history, turn = [], 0

    while True:
        turn += 1
        print(f"\nTurn {turn} – |L| = {len(candidates)}")

    
        if len(candidates) == 1:                       # only one possible word
            guess, exp_len = candidates[0], 1.0
        else:
            guess, exp_len = best_guess_by_expected_candidates(
                candidates, guess_pool)

        print(f"Chosen guess '{guess}' (E[|L'|] ≈ {exp_len:.2f})")

        fb = feedback(guess, target)
        print(f"Feedback: {fb}")
        history.append((turn, guess, fb, exp_len))

        if fb == "G" * len(target):
            print(f"Solved in {turn} guesses")
            break

        candidates = filter_candidates(candidates, guess, fb)
        if not candidates:
            print("No candidates remain – something went wrong")
            break

    return history

if __name__ == "__main__":
    for target_word in ['award','forgo', 'reach']:
        word_list   = load_words_from_file("wordle_dict.txt")  # solution list
        # word_list   = load_words_from_file("dict_test.txt")  # solution list
        # If you have a larger 'allowed guesses' file, load it and pass as guess_pool.
        simulate_game(target_word, word_list, word_list)
