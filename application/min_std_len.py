from utils.instance_utils import *
from utils.wordle_tools_utils import *
import numpy as np
import sys


def main():
    instance = get_instance()
    
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance

    best_w = None
    best_balance_score = float('inf')  # smaller is better
    best_lengths = []

    for i, w in enumerate(all_words):
        sys.stdout.write(f"word {i+1}/{len(all_words)}\r")

        lengths = []
        for target_word in key_words:
            guess_results = get_guess_results(target_word, w)
            filtered = filter_words(key_words, guess_results)
            lengths.append(len(filtered))
        
        # Measure imbalance — use standard deviation or range
        balance_score = np.std(lengths)  # could also use: max(lengths) - min(lengths)
        
        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_w = w
            best_lengths = lengths

    print("Best word:", best_w)
    print("Filtered lengths:", best_lengths)
    print("Balance score (std):", best_balance_score)


if __name__ == "__main__":
    main()