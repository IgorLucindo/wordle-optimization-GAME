from collections import Counter
from cupyx import scatter_add
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
        (True,  True):  _get_best_guess_composite2_GPU,
    }
    return mapping[(configs['GPU'], configs['composite_score'])]


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns (CPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]
    
    scores = np.zeros(len(G), dtype=int)

    for i, g in enumerate(G):
        P_g = np.unique(F[T, g])
        
        # Cardinality of distinct values
        scores[i] = len(P_g)

    # Get best guess
    g_star = G[np.argmax(scores)]

    return g_star


def _get_best_guess_GPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns
    GPU Parallelized version
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]  # shape: (len(T), len(G))

    # Flatten in column-major order to align with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # Scatter pattern occurrences
    pattern_seen = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(pattern_seen, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))

    # Count unique feedbacks per guess
    scores = (pattern_seen > 0).sum(axis=1)

    # --- Pruning ---
    candidate_scores = scores[T]
    mask = candidate_scores == n
    if cp.any(mask):
        return T[cp.argmax(mask)]

    # --- Best guess ---
    g_star = G[cp.argmax(scores)]
    return g_star


def _get_best_guess_composite_CPU(T, G, F, _lambda=2):
    """
    Finds the best guess using a composite score (CPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]
    
    scores = np.zeros(len(G), dtype=float)

    for i, g in enumerate(G):
        col_values = F[T, g]
        P_g = np.unique(col_values)
        C = Counter(col_values)

        # Multiset of partition sizes
        S_g = [C[p] for p in P_g]

        # std of partition sizes
        sigma_g = np.std(S_g)
        
        # Composite score
        scores[i] = len(P_g) - _lambda * sigma_g

    # Get best guess
    g_star = G[np.argmax(scores)]

    return g_star


def _get_best_guess_composite_GPU(T, G, F, _lambda=2):
    """
    Finds the best guess using a composite score (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # Number of possible feedback patterns

    # Extract feedback submatrix for feasible targets vs all guesses
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|)

    # Flatten in column-major order to align correctly with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # --- 1️⃣ Build histogram table [guess × feedback] ---
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))

    # --- 2️⃣ Compute per-guess statistics ---
    # How many distinct feedbacks each guess produces
    num_feedbacks = (hist > 0).sum(axis=1)

    # Counts of each feedback pattern per guess
    sum_counts = hist.sum(axis=1)
    sum_counts_sq = (hist ** 2).sum(axis=1)

    mean_counts = sum_counts / cp.maximum(num_feedbacks, 1)
    mean_counts_sq = sum_counts_sq / cp.maximum(num_feedbacks, 1)
    std_partition = cp.sqrt(cp.maximum(mean_counts_sq - mean_counts**2, 0))

    # --- 3️⃣ Composite score ---
    scores = num_feedbacks - _lambda * std_partition

    # --- 4️⃣ Pruning ---
    candidate_scores = scores[T]
    mask = candidate_scores == n
    if cp.any(mask):
        return T[cp.argmax(mask)]

    # # --- 5️⃣ Best guess ---
    g_star = G[cp.argmax(scores)]
    return g_star


def _get_best_guess_composite2_GPU(T, G, F, _lambda=2):
    """
    Finds the best guess using a composite score (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # Number of possible feedback patterns

    # Extract feedback submatrix for feasible targets vs all guesses
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|)

    # Flatten in column-major order to align correctly with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # --- 1️⃣ Build histogram table [guess × feedback] ---
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))

    # --- 2️⃣ Compute per-guess statistics ---
    num_feedbacks = (hist > 0).sum(axis=1)
    sum_counts = hist.sum(axis=1)
    sum_counts_sq = (hist ** 2).sum(axis=1)
    mean_counts = sum_counts / cp.maximum(num_feedbacks, 1)
    mean_counts_sq = sum_counts_sq / cp.maximum(num_feedbacks, 1)
    std_partition = cp.sqrt(cp.maximum(mean_counts_sq - mean_counts**2, 0))
    max_group_sizes = hist.max(axis=1)

    tie_breaker_bonus = cp.zeros(nG, dtype=cp.float32)
    tie_breaker_bonus[T] = 1

    # --- 3️⃣ Composite score ---
    # scores = mean_counts + 0.02*max_group_sizes + 0.01*std_partition - 0.1*tie_breaker_bonus # (minimize)
    scores = num_feedbacks - 0.1*max_group_sizes - 0.1*std_partition + 0.5*tie_breaker_bonus # (maximize)

    # # --- 5️⃣ Best guess ---
    g_star = G[cp.argmax(scores)]
    return g_star