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


def _get_best_guess_CPU(T, F):
    """
    """
    pass


def _get_best_guess_GPU(T, F):
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

    # Pick best guess
    best_w = int(cp.argmax(scores).item())

    return best_w


def _get_best_guess_composite_CPU(T, F, _lambda=2):
    """
    """
    pass


def _get_best_guess_composite_GPU(T, F, _lambda=2):
    """
    Finds the best guess using a composite score
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

    # Pick best guess
    best_w = int(cp.argmax(scores).item())

    return best_w