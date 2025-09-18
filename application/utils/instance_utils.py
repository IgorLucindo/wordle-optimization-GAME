from utils.guess_selection_utils import *
import cupy as cp
import json
import ast


def get_instance(configs):
    """
    Return word lists from dataset of words, feedback matrix and best guess function
    """
    T = _get_words("dataset/solutions.txt") # Target words
    G = T + _get_words("dataset/non_solutions.txt") # Guesses
    F = _get_feedback_matrix(T, G, configs['GPU'])
    get_best_guess = best_guess_function(configs)

    return G, T, F, get_best_guess


def get_guess_tree():
    """
    Loads the decision tree from a JSON file
    """
    with open('dataset/guess_tree.json', 'r') as f:
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


def _get_feedback_matrix(T, G, gpu_flag):
    """
    """
    if gpu_flag:
        return get_feedback_matrix_GPU(T, G)
    else:
        return get_feedback_matrix_CPU(T, G)


def get_feedback_matrix_CPU(key_words_str, all_words_str):
    """
    """
    pass


def get_feedback_matrix_GPU(key_words_str, all_words_str):
    """
    Return feedback matrix of shape (T, G) with dtype=cp.uint8, where matrix[i, j] is the
    base-3 encoded feedback code for key_words[i] as target and all_words[j] as guess.
    Assumes key_words and all_words are pre-encoded CuPy arrays with values 0-25 for letters a-z.
    """
    # Create staks with encoded words
    key_words = cp.stack([_encode_word(w) for w in key_words_str])
    all_words = cp.stack([_encode_word(w) for w in all_words_str])
    
    K = key_words.shape[0]
    G = all_words.shape[0]
    L = key_words.shape[1]  # Length of each word

    # Compute letter counts for each target
    flat_idx = cp.ravel(key_words) + cp.repeat(cp.arange(K), L) * 26
    count_t = cp.bincount(flat_idx, minlength=K * 26).reshape(K, 26).astype(cp.int32)

    # Compute equality mask for greens
    equal = key_words[:, None, :] == all_words[None, :, :]

    # Initialize feedback
    feedback = cp.zeros((K, G, L), dtype=cp.uint8)
    feedback[equal] = 2

    # Compute green counts per letter per pair
    green_counts = cp.zeros((K, G, 26), dtype=cp.int32)
    for l in range(L):
        mask_l = equal[:, :, l]
        k, g = cp.where(mask_l)
        if len(k):
            c = key_words[k, l]
            green_counts[k, g, c] += 1

    # Remaining counts after greens
    remaining_counts = count_t[:, None, :] - green_counts

    # Second pass for yellows
    for i in range(L):
        mask = (feedback[:, :, i] == 0)
        letter_i = all_words[:, i]
        idx_k = cp.arange(K)[:, None]
        idx_g = cp.arange(G)[None, :]
        idx_c = letter_i[None, :]
        rem = remaining_counts[idx_k, idx_g, idx_c]
        cond = (rem > 0) & mask
        feedback[:, :, i][cond] = 1
        remaining_counts[idx_k, idx_g, idx_c] -= cond.astype(cp.int32)

    # Compute the encoded feedback codes
    powers = cp.power(3, cp.arange(L - 1, -1, -1), dtype=cp.uint8)
    code = cp.sum(feedback * powers[None, None, :], axis=2, dtype=cp.uint8)
    return code


def _encode_word(word):
    return cp.array([ord(c) - 97 for c in word], dtype=cp.int8)