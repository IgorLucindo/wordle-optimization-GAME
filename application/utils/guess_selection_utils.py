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
        (True,  True):  _get_best_guess_composite_GPU,
    }
    return mapping[(configs['GPU'], configs['composite_score'])]


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns (CPU)
    """
    n = len(T)
    if n == 1:
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


def _get_best_guess_composite_GPU(T, G, F, _lambda=1):
    """
    Finds the best guess using a composite score (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # number of possible feedback patterns

    # Extract feedback submatrix for feasible targets vs all guesses
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|)

    # Flatten in column-major order to align correctly with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # --- 1️⃣ Build histogram table [guess × feedback] ---
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))

    # --- 2️⃣ Compute per-guess statistics ---
    # how many distinct feedbacks each guess produces
    num_feedbacks = (hist > 0).sum(axis=1)

    # counts of each feedback pattern per guess
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
    g_star = cp.argmax(scores)
    return g_star


def _get_best_guess_2steps_GPU(T, G, F):
    """
    Two-step lookahead Wordle guess selection (GPU version)
    Minimizes expected feasible set size after up to two guesses.
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # total feedback patterns

    # --- Precompute feedback submatrix for T × G ---
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|)
    best_scores = cp.zeros(nG, dtype=cp.float32)
    num_feedbacks = cp.zeros(nG, dtype=cp.float32)

    # --- Preallocate for scatter_add path ---
    pattern_seen = cp.zeros((nG, base), dtype=cp.int32)

    for i, g1 in enumerate(G):
        fb_g1 = feedbacks_sub[:, i]
        unique_fb, inverse = cp.unique(fb_g1, return_inverse=True)
        num_feedbacks[i] = len(unique_fb)

        total_expectation = 0.0

        for j, fb in enumerate(unique_fb):
            # Case: solved on first guess
            if fb == 242:
                continue

            # Feasible subset after feedback j
            T_i_idx = cp.nonzero(inverse == j)[0]
            m = len(T_i_idx)
            if m == 1:
                total_expectation += 1.0 * (m / n)
                continue

            # --- Step 2: lookahead for best g₂ ---
            fb_sub = feedbacks_sub[T_i_idx[:, None], G]  # shape: (m, nG)
            flat_fb = fb_sub.ravel(order="F")
            flat_col = cp.repeat(cp.arange(nG), m)
            pattern_seen.fill(0)
            scatter_add(pattern_seen, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))

            # Distinct feedbacks per guess (diversity)
            num_fb_g2 = (pattern_seen > 0).sum(axis=1).astype(cp.float32)
            mask_g2 = cp.isin(G, T_i_idx).astype(cp.float32)

            # Expected feasible size for g₂
            exp_size_g2 = (m - mask_g2) / (num_fb_g2 - mask_g2 + 1e-12)
            scores_g2 = exp_size_g2 + 1e-3 * (num_fb_g2 - mask_g2)
            best_g2_score = scores_g2.min()  # best g₂ for this feedback

            total_expectation += best_g2_score * (m / n)

        best_scores[i] = total_expectation

    # --- g₁ scoring using your heuristic ---
    mask_g1 = cp.isin(G, T).astype(cp.float32)
    exp_size_g1 = (n - mask_g1) / (num_feedbacks - mask_g1 + 1e-12)
    scores_g1 = best_scores + exp_size_g1 + 1e-3 * (num_feedbacks - mask_g1)

    g_star = G[cp.argmin(scores_g1)]
    return g_star