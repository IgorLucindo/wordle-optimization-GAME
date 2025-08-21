from utils.wordle_tools_utils import *
from utils.calculation_utils import *
from utils.hash_utils import *
import cupy as cp
import json
import ast


def get_instance():
    """
    Return instance for wordle solver model
    """
    key_words = _get_words("dataset/solutions.txt")
    all_words = key_words + _get_words("dataset/non_solutions.txt")
    words = all_words.copy()
    num_of_letters = len(words[0])
    num_of_attempts = 6
    tree = _get_decision_tree()

    return all_words, words, key_words, num_of_letters, num_of_attempts, tree


def fiter_instance(instance, word_guess, feedback):
    """
    Return filtered instances given guess results
    """
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance
    words = filter_words(words, word_guess, feedback)
    key_words = filter_words(key_words, word_guess, feedback)

    return all_words, words, key_words, num_of_letters, num_of_attempts, tree


def _get_decision_tree():
    with open('dataset/decision_tree.json', 'r') as f:
        tree = json.load(f)
        tree['nodes'] = {ast.literal_eval(k): v for k, v in tree['nodes'].items()}

    return tree


def _get_words(filepath):
    words = []

    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    return words


def get_feedback(target, guess):
    """
    Compute feedback for a single target and guess, returning a uint8 base-3 encoded code.
    target, guess: CuPy arrays of shape (L,) with values 0-25 (a-z).
    """
    L = target.shape[0]
    feedback = cp.zeros(L, dtype=cp.uint8)  # 0=B, 1=Y, 2=G

    # Letter counts
    counts = cp.bincount(target, minlength=26)

    # First pass: greens
    equal = (target == guess)
    feedback[equal] = 2
    counts = counts.astype(cp.int32)  # Ensure int32 for safe subtraction
    counts[target[equal]] -= 1

    # Second pass: yellows
    for i in range(L):
        if feedback[i] == 0:
            idx = guess[i]
            if counts[idx] > 0:
                feedback[i] = 1
                counts[idx] -= 1

    # Encode as uint8 (base-3)
    powers = cp.power(3, cp.arange(L - 1, -1, -1), dtype=cp.uint8)
    code = cp.sum(feedback * powers, dtype=cp.uint8)
    return code


def get_feedback_matrix(key_words_str, all_words_str):
    """
    Return feedback matrix of shape (T, G) with dtype=cp.uint8, where matrix[i, j] is the
    base-3 encoded feedback code for key_words[i] as target and all_words[j] as guess.
    Assumes key_words and all_words are pre-encoded CuPy arrays with values 0-25 for letters a-z.
    """
    # Create staks with encoded words
    key_words = cp.stack([encode_word(w) for w in key_words_str])
    all_words = cp.stack([encode_word(w) for w in all_words_str])
    
    T = key_words.shape[0]
    G = all_words.shape[0]
    if T == 0 or G == 0:
        return cp.array([], dtype=cp.uint8)
    L = key_words.shape[1]  # Length of each word

    # Compute letter counts for each target
    flat_idx = cp.ravel(key_words) + cp.repeat(cp.arange(T), L) * 26
    count_t = cp.bincount(flat_idx, minlength=T * 26).reshape(T, 26).astype(cp.int32)

    # Compute equality mask for greens
    equal = key_words[:, None, :] == all_words[None, :, :]

    # Initialize feedback
    feedback = cp.zeros((T, G, L), dtype=cp.uint8)
    feedback[equal] = 2

    # Compute green counts per letter per pair
    green_counts = cp.zeros((T, G, 26), dtype=cp.int32)
    for l in range(L):
        mask_l = equal[:, :, l]
        t, g = cp.where(mask_l)
        if len(t):
            c = key_words[t, l]
            green_counts[t, g, c] += 1

    # Remaining counts after greens
    remaining_counts = count_t[:, None, :] - green_counts

    # Second pass for yellows
    for i in range(L):
        mask = (feedback[:, :, i] == 0)
        letter_i = all_words[:, i]
        idx_t = cp.arange(T)[:, None]
        idx_g = cp.arange(G)[None, :]
        idx_c = letter_i[None, :]
        rem = remaining_counts[idx_t, idx_g, idx_c]
        cond = (rem > 0) & mask
        feedback[:, :, i][cond] = 1
        remaining_counts[idx_t, idx_g, idx_c] -= cond.astype(cp.int32)

    # Compute the encoded feedback codes
    powers = cp.power(3, cp.arange(L - 1, -1, -1), dtype=cp.uint8)
    code = cp.sum(feedback * powers[None, None, :], axis=2, dtype=cp.uint8)
    return code