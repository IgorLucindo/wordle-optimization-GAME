from utils.guess_selection_utils import *
import numpy as np
import cupy as cp


def get_instance(flags, configs):
    """
    Return word lists from dataset of words, feedback matrix and best guess function
    """
    T = _get_words("dataset/solutions.txt") # Target words
    G = T + _get_words("dataset/non_solutions.txt") # Guesses
    F = _get_feedback_matrix(T, G, configs)
    C = _get_feedback_compatibility_matrix(configs)
    decode_feedback = decode_feedback_GPU if configs['GPU'] else decode_feedback_CPU
    instance_data = (G, T, F, C, decode_feedback)
    get_best_guess = best_guess_function(instance_data, flags, configs)
    get_best_guesses = best_guesses_function(configs)

    return instance_data + (get_best_guess, get_best_guesses,)


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
        (False, False): _get_feedback_matrix_CPU,
        (True,  False): _get_feedback_matrix_GPU,
        (False, True):  _get_feedback_matrix_hardmode_CPU,
        (True,  True):  _get_feedback_matrix_hardmode_GPU_batched,
    }
    return mapping[(configs['GPU'], configs['hard_mode'])](T, G)


def _get_feedback_matrix_CPU(T, G):
    """
    Returns the encoded feedback matrix, where F[t, g] represents the
    feedback pattern of target t and guess g (CPU)
    """
    K = len(T)
    nG = len(G)
    L = 5
    
    # Pre-encode words to integers (0-25) for easier indexing
    T_int = [[ord(c) - 97 for c in w] for w in T]
    G_int = [[ord(c) - 97 for c in w] for w in G]
    
    # Initialize output matrix
    # F = Matrix of shape (T, G)
    F = np.zeros((K, nG), dtype=np.uint8)

    # Powers of 3 for encoding [3^4, 3^3, ..., 3^0]
    powers = [3**(L-1-i) for i in range(L)]

    # line 1: Precompute c_t (letter counts for targets)
    # c_t[t][char] = count
    c_t = np.zeros((K, 26), dtype=np.int8)
    for t_idx, t_word in enumerate(T_int):
        for char in t_word:
            c_t[t_idx][char] += 1

    # line 3: Loop T
    for t_idx in range(K):
        target = T_int[t_idx]
        target_counts = c_t[t_idx] # This is c_t from line 1
        
        # line 4: Loop G
        for g_idx in range(nG):
            guess = G_int[g_idx]
            
            # line 2: f_tgi <- 0 (Initialize row for this pair)
            f_row = [0] * L
            
            # line 5: c' <- c_t (Copy counts)
            c_prime = target_counts.copy()
            
            # line 6: Loop i (Green pass)
            for i in range(L):
                g_char = guess[i]
                t_char = target[i]
                
                # line 7: If g_i = t_i
                if g_char == t_char:
                    f_row[i] = 2             # Green
                    c_prime[g_char] -= 1     # Decrement count
            
            # line 10: Loop i (Yellow pass)
            for i in range(L):
                g_char = guess[i]
                
                # line 11: If f_tgi = 0 AND c'(g_i) > 0
                if f_row[i] == 0 and c_prime[g_char] > 0:
                    f_row[i] = 1             # Yellow
                    c_prime[g_char] -= 1     # Decrement count
            
            # line 15: Encode row to single integer
            # F_tg <- (f_tg1, ..., f_tgl)
            code = 0
            for i in range(L):
                code += f_row[i] * powers[i]
            
            F[t_idx, g_idx] = code

    return F


def _get_feedback_matrix_GPU(T, G):
    """
    Returns the encoded feedback matrix, where F[t, g] represents the
    feedback pattern of target t and guess g (GPU)
    """
    # Create staks with encoded words
    key_words = cp.stack([_encode_word(w) for w in T])
    all_words = cp.stack([_encode_word(w) for w in G])
    
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
    F = cp.sum(feedback * powers[None, None, :], axis=2, dtype=cp.uint8)
    return F


def _get_feedback_matrix_hardmode_CPU(T, G):
    """
    Returns the encoded feedback matrix, where F[t, g] represents the
    feedback pattern of target t and guess g, hardmode version (CPU)
    """
    return _get_feedback_matrix_CPU(G, G)


def _get_feedback_matrix_hardmode_GPU_batched(T, G, batch_size=1000):
    """
    Returns the encoded feedback matrix, where F[t, g] represents the
    feedback pattern of target t and guess g, hardmode batched version (GPU)
    """
    nG = len(G)
    F = cp.empty((nG, nG), dtype=cp.uint16)
    for i in range(0, nG, batch_size):
        end = min(i + batch_size, nG)
        F[i:end] = _get_feedback_matrix_GPU(G[i:end], G)
        cp._default_memory_pool.free_all_blocks()
    return F


def _get_feedback_compatibility_matrix(configs, l=5):
    """
    Return a compatibility table for feedback codes in hard mode
    """
    if not configs['hard_mode']:
        return None
    
    xp = cp if configs['GPU'] else np
    n = 243
    codes = xp.arange(n, dtype=xp.int32)
    digits = ((codes[:, None] // (3 ** xp.arange(l-1, -1, -1))) % 3).astype(xp.int8)

    # Compare all pairs (i, j): we want i >= j elementwise
    C = xp.all(digits[:, None, :] >= digits[None, :, :], axis=2)

    return C


def _encode_word(word):
    return cp.array([ord(c) - 97 for c in word], dtype=cp.int8)


def decode_feedback_CPU(f):
    base = 3
    L = 5
    exps = np.power(base, np.arange(L - 1, -1, -1))
    return (f // exps) % base


def decode_feedback_GPU(f):
    base = 3
    L = 5
    exps = cp.power(base, cp.arange(L - 1, -1, -1))
    return (f // exps) % base