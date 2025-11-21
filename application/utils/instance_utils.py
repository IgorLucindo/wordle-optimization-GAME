from utils.guess_selection_utils import *
import numpy as np
import cupy as cp
import json
import ast


def get_instance(configs):
    """
    Return word lists from dataset of words, feedback matrix and best guess function
    """
    T = _get_words("dataset/solutions.txt") # Target words
    G = T + _get_words("dataset/non_solutions.txt") # Guesses
    F = _get_feedback_matrix(T, G, configs)
    C = _get_feedback_compatibility_matrix(configs)
    get_best_guess = best_guess_function(configs)
    best_first_guesses = _get_best_first_guesses(configs)

    return G, T, F, C, get_best_guess, best_first_guesses


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


def _get_feedback_matrix(T, G, configs):
    """
    Returns the feedback matrix based on the GPU and hard_mode configs
    """
    mapping = {
        (False, False): get_feedback_matrix_CPU,
        (True,  False): get_feedback_matrix_GPU,
        (False, True):  _get_feedback_matrix_GPU_batched,
        (True,  True):  _get_feedback_matrix_GPU_batched,
    }
    return mapping[(configs['GPU'], configs['hard_mode'])](T, G)


def get_feedback_matrix_CPU(key_words_str, all_words_str):
    """
    """
    F = get_feedback_matrix_GPU(key_words_str, all_words_str)
    return F.get()


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


def _get_feedback_matrix_GPU_batched(T, G, batch_size=1000):
    """
    Return feedback matrix of shape (G, G) with dtype=cp.uint8, where matrix[i, j] is the
    base-3 encoded feedback code for G[i] as target and G[j] as guess.
    This version processes the target words in batches to manage memory on the GPU.
    """
    nG = len(G)
    F = cp.empty((nG, nG), dtype=cp.uint16)
    for i in range(0, nG, batch_size):
        end = min(i + batch_size, nG)
        F[i:end] = get_feedback_matrix_GPU(G[i:end], G)
        cp._default_memory_pool.free_all_blocks()
    return F


def _get_feedback_compatibility_matrix(configs, l=5):
    """
    Return a compatibility table for feedback codes in hard mode.
    A feedback code `fb1` is compatible with `fb2` if `fb1` could be generated
    from the same target word as `fb2`, given that `fb2` was the actual feedback.
    This means that for each position, the feedback for `fb1` must be "at least as good"
    as the feedback for `fb2`.
    """
    if not configs['hard_mode']:
        return None
    
    xp = cp if configs['GPU'] else np
    n = 243
    codes = xp.arange(n, dtype=xp.int32)
    digits = ((codes[:, None] // (3 ** xp.arange(l-1, -1, -1))) % 3).astype(xp.int8)

    # Compare all pairs (i,j): we want j >= i elementwise
    # Expand dims to (243, 1, 5) and (1, 243, 5)
    feedback_compat = xp.all(digits[None, :, :] >= digits[:, None, :], axis=2)

    return feedback_compat


def _get_best_first_guesses(configs):
    """
    Return n best first guesses
    """
    n = configs['#trees']
    best_first_guesses = ['salet', 'crane', 'reast', 'crate', 'aback', 'trace', 'slate', 'carle', 'slane', 'slant', 'trice', 'torse', 'carte', 'least', 'rance', 'trine', 'stale', 'train', 'prate', 'slart', 'roast', 'taser', 'caret', 'clast', 'earst', 'lance', 'trone', 'carse', 'stare', 'leant', 'react', 'toile', 'peart', 'roist', 'trade', 'drant', 'stane', 'saint', 'scale', 'crine', 'crone', 'trape', 'crise', 'clart', 'plate', 'roset', 'sorel', 'canst', 'dealt', 'loast', 'crost', 'raine', 'truce', 'parse', 'reist', 'resat', 'snirt', 'corse', 'close', 'riant', 'slice', 'alist', 'sault', 'prase', 'soare', 'store', 'caner', 'orant', 'liane', 'plane', 'tripe', 'tares', 'trail', 'tried', 'raise', 'stole', 'trans', 'roate', 'saner', 'snare', 'spalt', 'arose', 'cruet', 'palet', 'snore', 'antre', 'strae', 'artel', 'cline', 'clint', 'liart', 'orate', 'tears', 'cater', 'plast', 'scant', 'spart', 'stile', 'thale', 'aline']
    return best_first_guesses[:n]