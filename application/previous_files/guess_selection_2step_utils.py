from numba import cuda, uint8
from cupyx import scatter_add
import numpy as np
import cupy as cp


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


def _get_best_guess_2steps_composite_GPU(T, G, F, _lambda=5):
    """
    Two-step lookahead Wordle guess selection (GPU version with composite heuristic).
    Combines mean/std partition balance with expected feasible size minimization.
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # number of possible feedbacks

    # --- Precompute feedback submatrix ---
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|)
    best_scores = cp.zeros(nG, dtype=cp.float32)
    num_feedbacks = cp.zeros(nG, dtype=cp.float32)
    pattern_seen = cp.zeros((nG, base), dtype=cp.int32)

    for i, g1 in enumerate(G):
        fb_g1 = feedbacks_sub[:, i]
        unique_fb, inverse = cp.unique(fb_g1, return_inverse=True)
        num_feedbacks[i] = len(unique_fb)
        total_expectation = 0.0

        for j, fb in enumerate(unique_fb):
            if fb == 242:
                continue

            # Feasible subset after this feedback
            T_i_idx = cp.nonzero(inverse == j)[0]
            m = len(T_i_idx)
            if m <= 1:
                total_expectation += (m / n)
                continue

            # --- Step 2: candidate feedbacks for second guess ---
            fb_sub = feedbacks_sub[T_i_idx[:, None], G]  # (m, nG)
            flat_fb = fb_sub.ravel(order="F")
            flat_col = cp.repeat(cp.arange(nG), m)

            pattern_seen.fill(0)
            scatter_add(pattern_seen, (flat_col, flat_fb),
                        cp.ones_like(flat_fb, dtype=cp.int32))

            # --- Stats for each g₂ ---
            num_fb_g2 = (pattern_seen > 0).sum(axis=1).astype(cp.float32)
            sum_counts = pattern_seen.sum(axis=1).astype(cp.float32)
            sum_counts_sq = (pattern_seen ** 2).sum(axis=1).astype(cp.float32)

            mean_counts = sum_counts / cp.maximum(num_fb_g2, 1)
            mean_counts_sq = sum_counts_sq / cp.maximum(num_fb_g2, 1)
            std_partition = cp.sqrt(cp.maximum(mean_counts_sq - mean_counts ** 2, 0))

            # Composite + tie-breaker
            mask_g2 = cp.isin(G, T_i_idx).astype(cp.float32)
            exp_size_g2 = (m - mask_g2) / (num_fb_g2 - mask_g2 + 1e-12)

            # Blend expected size + composite heuristic
            # Lower is better
            scores_g2 = exp_size_g2 + _lambda * std_partition - 1e-3 * num_fb_g2

            best_g2_score = scores_g2.min()
            total_expectation += best_g2_score * (m / n)

        best_scores[i] = total_expectation

    # --- Final g₁ selection with expected-size heuristic ---
    mask_g1 = cp.isin(G, T).astype(cp.float32)
    exp_size_g1 = (n - mask_g1) / (num_feedbacks - mask_g1 + 1e-12)
    scores_g1 = best_scores + exp_size_g1 + 1e-3 * (num_feedbacks - mask_g1)

    g_star = G[cp.argmin(scores_g1)]
    return g_star


@cuda.jit
def partition_guess_unique_counts_kernel(feedbacks_all, boundaries, counts):
    """
    For each (partition, guess), count unique feedbacks in that partition.
    """
    idx = cuda.grid(1)
    m = boundaries.shape[0] - 1
    num_guesses = feedbacks_all.shape[1]
    total_threads = m * num_guesses
    if idx >= total_threads:
        return

    part = idx // num_guesses
    g = idx % num_guesses

    start = boundaries[part]
    end = boundaries[part + 1]
    size = end - start

    if size <= 0:
        counts[part, g] = 0
        return
    if size == 1:
        v = feedbacks_all[start, g]
        counts[part, g] = 1 if v != 255 else 0
        return

    seen = cuda.local.array(243, uint8)
    for k in range(243):
        seen[k] = 0

    for r in range(start, end):
        val = feedbacks_all[r, g]
        if val != 255:
            seen[val] = 1

    c = 0
    for k in range(243):
        c += seen[k]
    counts[part, g] = c


def _get_best_guess_2steps_GPU_numba(T, G, F, threads_per_block=2):
    """
    Optimized two-step GPU hybrid:
    - Minimizes data transfers and memory churn
    - Same logic, faster execution
    """
    n = int(len(T))
    if n <= 2:
        return T[0]

    feedbacks_sub = F[T[:, None], G]
    nG = len(G)
    base = 243

    best_scores = cp.zeros(nG, dtype=cp.float32)
    num_feedbacks = cp.zeros(nG, dtype=cp.float32)

    # Reuse host buffers to avoid reallocations each iteration
    counts_host = None
    d_counts = None

    for i in range(nG):
        fb_g1 = feedbacks_sub[:, i]
        unique_fb, inv_idx = cp.unique(fb_g1, return_inverse=True)
        m = len(unique_fb)
        num_feedbacks[i] = m
        if m == 0:
            continue

        # Sort indices for contiguous partitions
        order = cp.argsort(inv_idx)
        inv_sorted = inv_idx[order]
        T_sorted = T[order]

        inv_sorted_np = cp.asnumpy(inv_sorted)
        ends = np.searchsorted(inv_sorted_np, np.arange(1, m + 1), side='left').astype(np.int32)
        boundaries_host = np.empty(m + 1, dtype=np.int32)
        boundaries_host[0] = 0
        boundaries_host[1:] = ends

        # Get feedbacks_all as contiguous np array (uint8)
        feedbacks_all_np = cp.asnumpy(F[T_sorted[:, None], G])

        # Move data to device
        d_feedbacks_all = cuda.to_device(feedbacks_all_np)
        d_boundaries = cuda.to_device(boundaries_host)

        # Allocate counts once or reuse existing
        if counts_host is None or counts_host.shape != (m, nG):
            counts_host = np.zeros((m, nG), dtype=np.int32)
            d_counts = cuda.to_device(counts_host)
        else:
            d_counts.copy_to_device(counts_host)  # reset to zeros

        total_threads = m * nG
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        partition_guess_unique_counts_kernel[blocks, threads_per_block](
            d_feedbacks_all, d_boundaries, d_counts
        )
        cuda.synchronize()

        counts_host = d_counts.copy_to_host()

        # Expected feasible size heuristic (CPU part)
        total_expectation = 0.0
        G_np = G.get()
        T_sorted_np = T_sorted.get()

        for j in range(m):
            start = boundaries_host[j]
            end = boundaries_host[j + 1]
            m_j = end - start
            if m_j <= 1:
                total_expectation += (m_j / n)
                continue

            num_fb_g2 = counts_host[j, :].astype(np.float32)
            mask_g2 = np.isin(G_np, T_sorted_np[start:end]).astype(np.float32)
            exp_size_g2 = (m_j - mask_g2) / (num_fb_g2 - mask_g2 + 1e-12)
            scores_g2 = exp_size_g2 + 1e-3 * (num_fb_g2 - mask_g2)
            best_g2_score = np.min(scores_g2)
            total_expectation += best_g2_score * (m_j / n)

        best_scores[i] = total_expectation

        # Explicitly free only temporary arrays
        d_feedbacks_all = None
        d_boundaries = None

    # Final heuristic for g₁
    mask_g1 = cp.isin(G, T).astype(cp.float32)
    exp_size_g1 = (n - mask_g1) / (num_feedbacks - mask_g1 + 1e-12)
    scores_g1 = best_scores + exp_size_g1 + 1e-3 * (num_feedbacks - mask_g1)
    g_star = G[cp.argmin(scores_g1)]
    return g_star