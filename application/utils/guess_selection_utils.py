from classes.guess_tree import *
from utils.xp_utils import cp, HAS_CUPY
import numpy as np


def best_guess_functions(instance_data, is_target, flags, configs):
    """
    Selects the appropriate best guess functions.

    ``is_target`` is a boolean array of shape (n_G,) indicating which guesses
    are also valid terminal identification actions.
    """
    score_rule = configs.get('score', 'PC')

    def _get_best_guess_CPU(T, G, F):
        return _get_best_guess_CPU_impl(
            T, G, F, is_target=is_target,
            score_rule=score_rule, configs=configs)

    def _get_best_guess_GPU(T, G, F):
        return _get_best_guess_GPU_impl(
            T, G, F, is_target=is_target,
            score_rule=score_rule, configs=configs)

    _best_guess_functions = (_get_best_guess_CPU, _get_best_guess_GPU)
    _best_guesses_functions = best_guesses_functions(is_target, configs)

    # Subtree look-ahead wraps the avg-size rule.
    if configs['metric']:
        instance = instance_data + (_best_guess_functions, _best_guesses_functions)
        subtree = Guess_Tree(instance, flags, configs)

        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree, is_target, configs)
        _best_guess_functions = (get_best_guess_subtree, get_best_guess_subtree)

    return _best_guess_functions


def best_guesses_functions(is_target, configs):
    """Top-k candidate generators for Subtree-k."""
    score_rule = configs.get('score', 'PC')

    def get_best_guesses_CPU(T, G, F):
        return _get_best_guesses_CPU(T, G, F, num_of_guesses=configs['k'],
                                      is_target=is_target,
                                      score_rule=score_rule, configs=configs)

    def get_best_guesses_GPU(T, G, F):
        return _get_best_guesses_GPU(T, G, F, num_of_guesses=configs['k'],
                                      is_target=is_target,
                                      score_rule=score_rule, configs=configs)

    return (get_best_guesses_CPU, get_best_guesses_GPU)


def _get_best_guess_CPU_impl(T, G, F, is_target, score_rule='PC', configs=None):
    """
    Finds the best guess by minimizing the specified score rule (CPU).

    ``is_target`` is a boolean array indicating which guesses are valid
    terminal identification actions. For guessing games (Wordle/Mastermind),
    targets can self-identify; for sequential testing (Zoo), queries are not targets.
    ``score_rule`` specifies the scoring rule: 'PC' (Partition Count),
    'WA' (Weighted Average), or 'H' (Entropy).
    """
    base = configs.get('base', 243)
    n = len(T)

    # Shortcut: if few targets remain and first target can self-identify, return it
    if n <= configs.get('shortcut_threshold', 2):
        target = T[0]
        return target, is_target[int(target)]

    scores = np.empty(len(G), dtype=np.float64)

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]  # Keep only non-empty partitions
        indicator = 1 if is_target[g] else 0

        if score_rule == 'PC':
            # PC applies the indicator to the numerator (n - 1)
            scores[i] = (n - indicator) / len(partition_sizes)
        elif score_rule == 'WA':
            # WA applies the indicator subtraction to the sum of squares, divided by standard n
            scores[i] = (np.sum(partition_sizes ** 2) - indicator) / n
        elif score_rule == 'H':
            # Entropy MUST use standard n so probabilities sum to 1
            probs = partition_sizes / n
            entropy = -np.sum(probs * np.log2(probs))
            scores[i] = -entropy
        else:
            raise ValueError(f"Unknown score rule: {score_rule}")

    argmin = int(np.argmin(scores))
    g_star = G[argmin]
    is_target_flag = is_target[int(g_star)]
    return g_star, is_target_flag


def _get_best_guess_GPU_impl(T, G, F, is_target, score_rule='PC', configs=None):
    """
    Finds the best guess by minimizing the specified score rule (GPU).

    See ``_get_best_guess_CPU_impl`` for the semantics of
    ``is_target`` and ``score_rule``.
    """
    base = configs.get('base', 243)
    n = len(T)

    # Shortcut: if few targets remain and first target can self-identify, return it
    if n <= configs.get('shortcut_threshold', 2):
        target = T[0]
        return target, is_target[int(target)]

    nG = len(G)

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]

    # Linear Indexing
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    # Bincount over the flattened (target, guess) entries
    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    # Determine indicator function using is_target array
    indicator = is_target[G]

    if score_rule == 'PC':
        # PC: (n - indicator) / |P_g|
        num_feedbacks = (counts > 0).sum(axis=1)
        scores = (n - indicator) / num_feedbacks
    elif score_rule == 'WA':
        # WA: (sum of squares - indicator) / n
        scores = ((counts ** 2).sum(axis=1) - indicator) / n
    elif score_rule == 'H':
        # Entropy: uses standard n
        mask = counts > 0
        probs = counts / n  # Standard n, NOT n_adjusted
        log_probs = cp.where(mask, cp.log2(probs), 0.0)
        entropy = -cp.sum(probs * log_probs, axis=1)
        scores = -entropy
    else:
        raise ValueError(f"Unknown score rule: {score_rule}")

    # Best guess
    argmin = cp.argmin(scores)
    g_star = G[argmin]
    return g_star, indicator[argmin]


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10, is_target=None,
                          score_rule='PC', configs=None):
    """
    Finds the best guesses by minimizing the specified score rule (CPU).
    """
    base = configs.get('base', 243)
    n = len(T)
    scores = np.empty(len(G), dtype=np.float64)

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]

        n_adjusted = (n - 1) if is_target[g] else n

        if score_rule == 'PC':
            scores[i] = n_adjusted / len(partition_sizes)
        elif score_rule == 'WA':
            scores[i] = np.sum(partition_sizes ** 2) / n_adjusted
        elif score_rule == 'H':
            probs = partition_sizes / n_adjusted
            entropy = -np.sum(probs * np.log2(probs))
            scores[i] = -entropy
        else:
            raise ValueError(f"Unknown score rule: {score_rule}")

    sorted_indices = np.argsort(scores, kind='stable')
    g_star_idxs = sorted_indices[:num_of_guesses]
    picks = G[g_star_idxs]
    in_T = [is_target[int(g)] for g in picks]
    return picks, in_T


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10, is_target=None,
                          score_rule='PC', configs=None):
    """
    Finds the best guesses by minimizing the specified score rule (GPU).
    """
    base = configs.get('base', 243)
    n, nG = len(T), len(G)

    feedbacks_sub = F[T[:, None], G]
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    # Use is_target array directly
    in_T = is_target[G]
    n_adjusted = n - in_T

    if score_rule == 'PC':
        num_feedbacks = (counts > 0).sum(axis=1)
        scores = n_adjusted / num_feedbacks
    elif score_rule == 'WA':
        scores = (counts ** 2).sum(axis=1) / n_adjusted
    elif score_rule == 'H':
        mask = counts > 0
        probs = counts / n_adjusted[:, None]
        log_probs = cp.where(mask, cp.log2(probs), 0.0)
        entropy = -cp.sum(probs * log_probs, axis=1)
        scores = -entropy
    else:
        raise ValueError(f"Unknown score rule: {score_rule}")

    sorted_indices = cp.argsort(scores)
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs], in_T[g_star_idxs]


def _get_best_guess_subtree(T, G, F, subtree, is_target, configs):
    """
    Finds the best guess using a subtree metric
    """
    n = len(T)
    if n <= configs.get('shortcut_threshold', 2):
        target = T[0]
        return target, is_target[int(target)]

    T, G, xp, F, _, _, get_best_guesses = subtree.optimizer.get_context(T, G)

    if configs['metric'] == 1:
        G_prime, candidates_in_T = get_best_guesses(T, G, F)
    else:
        # Use is_target array for full evaluation
        G_prime = G
        candidates_in_T = is_target[G].tolist()

    g_star, is_target_flag = _get_best_subtree_candidate(T, G, G_prime, F, candidates_in_T, subtree)
    return g_star, is_target_flag


def _get_best_subtree_candidate(T, G, G_prime, F, candidates_in_T, subtree):
    """
    Evaluates the provided candidates (G_prime) using the subtree metric and selects the best one
    """
    subtree.T = T
    subtree.G = G
    scores = np.zeros(len(G_prime))

    for i, g_start in enumerate(G_prime):
        is_target_flag = candidates_in_T[i]
        D = subtree.build_subtree(g_start, is_target_flag)
        scores[i] = D.mean() + 0.001 * D.max()

    argmin = np.argmin(scores)
    g_star = G_prime[argmin]
    return g_star, candidates_in_T[argmin]
