from collections import Counter
import numpy as np
import cupy as cp


def best_guess_function(configs):
    """
    Selects the appropriate best guess function based on flags.
    """
    mapping = {
        (False, False): _get_best_guess_CPU,
        (True,  False): _get_best_guess_2steps_GPU,
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
    g_star = G[np.argmax(scores)]

    return g_star


def _get_best_guess_GPU(T, G, F):
    """
    Finds the best guess by maximizing the number of feedback patterns (GPU parallelization and prunning)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T[:, None], G]
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    unique_pairs = cp.unique(pairs)
    cols = unique_pairs // 243
    scores = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

    # Prunning: if a candidate in T achieves exactly len(T) feedback patterns, return that candidate
    candidate_scores = scores[T]
    mask = candidate_scores == n
    if mask.any():
        return T[cp.argmax(mask)]

    # Get best guess
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

    feedbacks_sub = F[T[:, None], G]
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
        return T[cp.argmax(mask)]

    # Get best guess
    g_star = G[cp.argmax(scores)]

    return g_star


def _get_best_guess_2steps_GPU1(T, G, F):
    """
    Finds the best guess by looking 2 steps ahead using GPU parallelization.

    Step 1: For each guess g in G, partition T by feedbacks.
    Step 2: For each resulting subset, compute the number of unique feedback patterns
            that *each possible next guess* could produce.
    The best guess maximizes the expected number of feedback patterns across 2 steps.
    """
    n = len(T)
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T[:, None], G]  # (|T|, |G|)
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    unique_pairs = cp.unique(pairs)
    cols = unique_pairs // 243
    scores_1step = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

    # --- 1-step pruning ---
    candidate_scores = scores_1step[T]
    mask = candidate_scores == n
    if mask.any():
        return T[cp.argmax(mask)]

    # --- 2-step lookahead ---
    # For each guess g, partition T by feedback
    best_scores = cp.zeros(len(G), dtype=cp.float32)

    for i, g in enumerate(G):
        # Feedbacks for this guess over T
        feedbacks_g = feedbacks_sub[:, i]
        unique_feedbacks, inverse_idx = cp.unique(feedbacks_g, return_inverse=True)
        m = len(unique_feedbacks)

        # For each partition (subset of T with same feedback)
        total = 0
        for j in range(m):
            T_j = T[inverse_idx == j]
            if len(T_j) <= 1:
                continue

            # Compute how well we can partition T_j using next guesses (parallel)
            feedbacks_j = F[T_j[:, None], G]
            pairs_j = feedbacks_j + cp.arange(feedbacks_j.shape[1]) * 243
            unique_pairs_j = cp.unique(pairs_j)
            cols_j = unique_pairs_j // 243
            score_j = cp.bincount(cols_j, minlength=feedbacks_j.shape[1])

            total += score_j.max()

        best_scores[i] = total / m if m > 0 else 0

    g_star = G[cp.argmax(best_scores)]
    return g_star


def _get_best_guess_2steps_GPU(T, G, F):
    n = len(T)
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T[:, None], G]
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    unique_pairs = cp.unique(pairs)
    cols = unique_pairs // 243
    scores_1step = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

    # Pruning
    candidate_scores = scores_1step[T]
    mask = candidate_scores == n
    if mask.any():
        return T[cp.argmax(mask)]

    # 2-step lookahead
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(len(G))]
    best_scores = cp.zeros(len(G), dtype=cp.float32)

    def eval_guess(i):
        with streams[i]:
            feedbacks_g = feedbacks_sub[:, i]
            unique_fb, inv_idx = cp.unique(feedbacks_g, return_inverse=True)
            total = 0
            for j, f_new in enumerate(unique_fb):
                T_j = T[inv_idx == j]
                if len(T_j) <= 1:
                    continue
                feedbacks_j = F[T_j[:, None], G]
                pairs_j = feedbacks_j + cp.arange(feedbacks_j.shape[1]) * 243
                unique_pairs_j = cp.unique(pairs_j)
                cols_j = unique_pairs_j // 243
                score_j = cp.bincount(cols_j, minlength=feedbacks_j.shape[1])
                total += score_j.max()
            best_scores[i] = total / len(unique_fb)

    # Launch all in parallel streams
    for i in range(len(G)):
        eval_guess(i)

    # Synchronize all
    for s in streams:
        s.synchronize()

    g_star = G[cp.argmax(best_scores)]
    return g_star