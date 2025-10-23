from collections import Counter
from numba import cuda, uint8
from cupyx import scatter_add
import numpy as np
import cupy as cp


def best_guess_function(configs):
    """
    Selects the appropriate best guess function based on flags.
    """
    mapping = {
        (False, False): _get_best_guess_CPU,
        # (True,  False): _get_best_guess_GPU,
        (True,  False): _get_best_guess_2steps_GPU,
        # (True,  False): _get_best_guess_2steps_GPU_numba,
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

    # --- 5️⃣ Best guess ---
    g_star = G[cp.argmax(scores)]
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
            fb_sub = feedbacks_sub[T_i_idx[:, None], G]
            pairs = fb_sub + cp.arange(nG) * base
            uniq_pairs = cp.unique(pairs)
            guess_idx = uniq_pairs // base
            feedback_count = cp.bincount(guess_idx, minlength=nG)

            # Distinct feedbacks per guess (diversity)
            num_fb_g2 = (feedback_count > 0).sum()
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


@cuda.jit(cache=True)
def partition_guess_unique_counts_kernel(feedbacks_all, boundaries, counts):
    # Linear thread index over (partition * num_guesses + guess)
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

    # Quick bail if tiny partition
    if end - start <= 0:
        counts[part, g] = 0
        return
    if end - start == 1:
        # single target -> only one feedback value
        v = feedbacks_all[start, g]
        counts[part, g] = 1 if v >= 0 else 0
        return

    # Local seen map for 243 possible feedbacks
    # NOTE: Numba requires compile-time constant size for local arrays
    seen = cuda.local.array(243, uint8)
    # initialize seen to zero
    for k in range(243):
        seen[k] = 0

    # mark seen feedbacks
    for r in range(start, end):
        val = feedbacks_all[r, g]  # uint8 in [0..242]
        # guard if val could be padding marker (we will make padding use 255)
        if val == 255:
            continue
        seen[int(val)] = 1

    # count seen
    c = 0
    for k in range(243):
        c += int(seen[k])
    counts[part, g] = c


def _get_best_guess_2steps_GPU_numba(T, G, F, threads_per_block=64):
    """
    Numba-CUDA accelerated 2-step lookahead.

    T : cupy array of target indices (1D)
    G : cupy array of guess indices (1D)
    F : cupy feedback matrix shape (num_targets, num_guesses) dtype=uint8
    """
    n = int(len(T))
    if n <= 2:
        return T[0]

    xp = cp
    # 1-step as before (do on GPU using CuPy)
    feedbacks_sub = F[T[:, None], G]  # shape (|T|, |G|), cupy array uint8
    pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
    unique_pairs = cp.unique(pairs)
    cols = unique_pairs // 243
    scores_1step = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

    # pruning
    candidate_scores = scores_1step[T]
    mask = candidate_scores == n
    if mask.any():
        return T[cp.argmax(mask)]

    # We'll evaluate each candidate guess g1, but heavy inner work is offloaded to the kernel.
    best_scores = cp.zeros(len(G), dtype=cp.float32)

    # Convert some things to host/numba-friendly arrays when needed
    # We'll still iterate over G (first-step guesses), but the expensive per-partition unique counts are on GPU via Numba.
    num_guesses = int(len(G))

    # Precompute full arange of second-guess indices as numpy for passing; but we will generate feedbacks_all per g1
    G_np_all = cp.asnumpy(G).astype(np.int32)

    for i in range(num_guesses):
        # For the i-th first guess, compute partitions of T by its feedbacks
        feedbacks_g = feedbacks_sub[:, i]  # cupy array (|T|,)
        unique_fb, inv_idx = cp.unique(feedbacks_g, return_inverse=True)
        m = int(len(unique_fb))
        if m == 0:
            best_scores[i] = 0.0
            continue

        # Sort inv_idx to get contiguous blocks
        order = cp.argsort(inv_idx)
        T_sorted = T[order]
        inv_sorted = inv_idx[order]

        # Build boundaries (m+1) using vectorized searchsorted: search values [1..m]
        # Convert inv_sorted to numpy for searchsorted with numpy (or use cp.searchsorted with cp.array)
        inv_sorted_np = cp.asnumpy(inv_sorted)
        search_vals = np.arange(1, m + 1, dtype=inv_sorted_np.dtype)
        ends = np.searchsorted(inv_sorted_np, search_vals, side='left').astype(np.int32)
        boundaries_host = np.empty(m + 1, dtype=np.int32)
        boundaries_host[0] = 0
        boundaries_host[1:] = ends  # boundaries_host[k] .. boundaries_host[k+1]-1 rows for partition k

        # Build feedbacks_all = F[T_sorted[:, None], G] but we need it as numpy then device
        # For performance we fetch feedbacks_all as cupy then convert once to numpy, then to numba device.
        # feedbacks_all shape (num_rows, num_guesses)
        feedbacks_all_cp = F[T_sorted[:, None], G]   # cupy
        # But we need dtype uint8 and padding marker for invalid rows; safe assumption F dtype is uint8
        feedbacks_all_np = cp.asnumpy(feedbacks_all_cp)  # numpy uint8, shape (num_rows, num_guesses)

        num_rows = feedbacks_all_np.shape[0]
        # Create device arrays for kernel
        d_feedbacks_all = cuda.to_device(feedbacks_all_np)           # device uint8 array (num_rows, num_guesses)
        d_boundaries = cuda.to_device(boundaries_host)              # device int32 array (m+1,)
        # counts: (m, num_guesses) int32
        counts_host = np.zeros((m, num_guesses), dtype=np.int32)
        d_counts = cuda.to_device(counts_host)

        # Launch kernel: total threads = m * num_guesses
        total_threads = m * num_guesses
        blocks = (total_threads + threads_per_block - 1) // threads_per_block

        partition_guess_unique_counts_kernel[blocks, threads_per_block](d_feedbacks_all, d_boundaries, d_counts)
        # sync
        cuda.synchronize()

        # Copy counts back to host (numpy) and reduce
        counts_host = d_counts.copy_to_host()  # shape (m, num_guesses)
        # For each partition choose maximum across guesses (axis=1? careful: counts are per (partition, second-guess))
        max_per_partition = counts_host.max(axis=1)  # shape (m,)
        # We sum maxima across partitions and divide by m to get score
        total = float(max_per_partition.sum())
        best_scores[i] = total / float(m)

        # free device arrays explicitly (help memory)
        d_feedbacks_all = None
        d_boundaries = None
        d_counts = None
        cp.get_default_memory_pool().free_all_blocks()

    g_star = G[cp.argmax(best_scores)]
    return g_star