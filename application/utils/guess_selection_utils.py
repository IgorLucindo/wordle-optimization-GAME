import numpy as np
import cupy as cp


def best_guess_function(configs):
    """
    Selects the appropriate best guess function based on flags.
    """
    mapping = {
        (False, False): _get_best_guess_CPU,
        (True,  False): _get_best_guess_GPU,
        (False, True):  _get_best_guess_composite_CPU,
        (True,  True):  _get_best_guess_composite_GPU,
    }
    return mapping[(configs['GPU'], configs['composite_score'])]


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns (CPU)
    """
    if len(T) == 1:
        return T[0]
    
    scores = np.zeros(len(G), dtype=int)

    for i, g in enumerate(G):
        P_g = np.unique(F[T, g])
        
        # Cardinality of distinct values
        scores[i] = len(P_g)

    # Get best guess
    g_star = np.argmax(scores)

    return g_star


def _get_best_guess_GPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T, :]
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    unique_pairs = cp.unique(pairs)
    cols = unique_pairs // 243
    scores = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

    # Prunning: if a candidate in T achieves exactly len(T) feedback patterns, return that candidate
    candidate_scores = scores[T]
    mask = candidate_scores == n
    if mask.any():
        return int(T[cp.argmax(mask)].item())

    # Get best guess
    g_star = int(cp.argmax(scores).item())

    return g_star


def _get_best_guess_composite_CPU(T, G, F, _lambda=1):
    """
    Finds the best guess using a composite score (CPU)
    """
    if len(T) == 1:
        return T[0]
    
    scores = np.zeros(len(G), dtype=float)

    for i, g in enumerate(G):
        col_values = F[T, g]
        P_g = np.unique(col_values)
        S_g = [len(T[col_values == p]) for p in P_g]

        # std of partition sizes
        sigma_g = np.std(S_g)
        
        # Composite score
        scores[i] = len(P_g) - _lambda * sigma_g

    # Get best guess
    g_star = np.argmax(scores)

    return g_star


def _get_best_guess_composite_GPU(T, G, F, _lambda=2):
    """
    Finds the best guess using a composite score (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T, :]
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    pairs_flat = pairs.ravel()
    uniq_pairs, counts = cp.unique(pairs_flat, return_counts=True)
    guess_idx = uniq_pairs // 243
    num_feedbacks = cp.bincount(guess_idx, minlength=feedbacks_sub.shape[1])
    sum_counts = cp.bincount(guess_idx, weights=counts, minlength=feedbacks_sub.shape[1])
    sum_counts_sq = cp.bincount(guess_idx, weights=counts**2, minlength=feedbacks_sub.shape[1])
    mean_counts = sum_counts / cp.maximum(num_feedbacks, 1)
    mean_counts_sq = sum_counts_sq / cp.maximum(num_feedbacks, 1)
    std_partition = cp.sqrt(cp.maximum(mean_counts_sq - mean_counts**2, 0))

    # Composite score 
    scores = num_feedbacks - _lambda*std_partition
    
    # Prunning: if a candidate in T achieves exactly len(T) feedback patterns, return that candidate
    candidate_scores = scores[T]
    mask = candidate_scores == n
    if mask.any():
        return int(T[cp.argmax(mask)].item())

    # Get best guess
    g_star = int(cp.argmax(scores).item())

    return g_star