from classes.guess_tree import *
from utils.xp_utils import cp, HAS_CUPY
import numpy as np


def best_guess_functions(instance_data, flags, configs):
    """
    Selects the appropriate best guess functions.

    ``is_target`` is a boolean array of shape (n_G,) indicating which guesses
    are also valid terminal identification actions.
    """
    score_rule = configs.get('score', 'PC')
    guesses_include_targets = bool(configs.get('guesses_include_targets', True))

    def _get_best_guess_CPU(T, G, F):
        return _get_best_guess_CPU_impl(
            T, G, F, guesses_include_targets=guesses_include_targets,
            score_rule=score_rule, configs=configs)

    def _get_best_guess_GPU(T, G, F):
        return _get_best_guess_GPU_impl(
            T, G, F, guesses_include_targets=guesses_include_targets,
            score_rule=score_rule, configs=configs)

    _best_guess_functions = (_get_best_guess_CPU, _get_best_guess_GPU)
    _best_guesses_functions = best_guesses_functions(configs)

    # Subtree look-ahead wraps the avg-size rule.
    if configs['metric']:
        instance = instance_data + (_best_guess_functions, _best_guesses_functions)
        subtree = Guess_Tree(instance, flags, configs)

        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree, configs)
        _best_guess_functions = (get_best_guess_subtree, get_best_guess_subtree)

    return _best_guess_functions


def best_guesses_functions(configs):
    """Top-k candidate generators for Subtree-k."""
    score_rule = configs.get('score', 'PC')

    def get_best_guesses_CPU(T, G, F):
        return _get_best_guesses_CPU(T, G, F, num_of_guesses=configs['k'],
                                      score_rule=score_rule, configs=configs)

    def get_best_guesses_GPU(T, G, F):
        return _get_best_guesses_GPU(T, G, F, num_of_guesses=configs['k'],
                                      score_rule=score_rule, configs=configs)

    return (get_best_guesses_CPU, get_best_guesses_GPU)


def _get_best_guess_CPU_impl(T, G, F, guesses_include_targets=True, score_rule='PC', configs=None):
    """
    Finds the best guess by minimizing the specified score rule (CPU).

    ``guesses_include_targets``: True if targets are valid guesses (Wordle/Mastermind),
    False if targets and guesses are disjoint (Zoo: targets=animals, guesses=attributes).
    ``score_rule``: 'PC' (Partition Count), 'WA' (Weighted Average), or 'H' (Entropy).
    """
    base = configs.get('base', 243)
    n = len(T)

    # Shortcut: return first target if few remain
    # Threshold: 2 if targets can be guessed directly, 1 (leaf only) otherwise
    shortcut_threshold = 2 if guesses_include_targets else 1
    if n <= shortcut_threshold:
        return T[0], guesses_include_targets

    scores = np.empty(len(G), dtype=np.float64)

    # Pre-compute indicator: True if guess is in current target set
    if guesses_include_targets:
        indicator = np.isin(G, T)
    else:
        indicator = np.zeros(len(G), dtype=bool)

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]

        if score_rule == 'PC':
            scores[i] = (n - indicator[i]) / len(partition_sizes)
        elif score_rule == 'WA':
            scores[i] = (np.sum(partition_sizes ** 2) - indicator[i]) / n
        elif score_rule == 'H':
            probs = partition_sizes / n
            entropy = -np.sum(probs * np.log2(probs))
            scores[i] = -entropy
        else:
            raise ValueError(f"Unknown score rule: {score_rule}")

    argmin = int(np.argmin(scores))
    g_star = G[argmin]
    return g_star, bool(indicator[argmin])


def _get_best_guess_GPU_impl(T, G, F, guesses_include_targets=True, score_rule='PC', configs=None):
    """
    Finds the best guess by minimizing the specified score rule (GPU).

    See ``_get_best_guess_CPU_impl`` for parameter semantics.
    """
    base = configs.get('base', 243)
    n = len(T)

    # Shortcut: return first target if few remain
    shortcut_threshold = 2 if guesses_include_targets else 1
    if n <= shortcut_threshold:
        return T[0], guesses_include_targets

    nG = len(G)

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]

    # Linear Indexing
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    # Bincount over the flattened (target, guess) entries
    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    # Pre-compute indicator
    if guesses_include_targets:
        indicator = cp.isin(G, T)
    else:
        indicator = cp.zeros(nG, dtype=bool)

    if score_rule == 'PC':
        num_feedbacks = (counts > 0).sum(axis=1)
        scores = (n - indicator) / num_feedbacks
    elif score_rule == 'WA':
        scores = ((counts ** 2).sum(axis=1) - indicator) / n
    elif score_rule == 'H':
        mask = counts > 0
        probs = counts / n
        log_probs = cp.where(mask, cp.log2(probs), 0.0)
        entropy = -cp.sum(probs * log_probs, axis=1)
        scores = -entropy
    else:
        raise ValueError(f"Unknown score rule: {score_rule}")

    argmin = cp.argmin(scores)
    g_star = G[argmin]
    return g_star, bool(indicator[argmin])


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10, score_rule='PC', configs=None):
    """
    Finds the top-k best guesses by minimizing the specified score rule (CPU).
    """
    base = configs.get('base', 243)
    n = len(T)
    guesses_include_targets = bool(configs.get('guesses_include_targets', True))
    scores = np.empty(len(G), dtype=np.float64)

    # Pre-compute indicator
    if guesses_include_targets:
        indicator = np.isin(G, T)
    else:
        indicator = np.zeros(len(G), dtype=bool)

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]

        if score_rule == 'PC':
            scores[i] = (n - indicator[i]) / len(partition_sizes)
        elif score_rule == 'WA':
            scores[i] = (np.sum(partition_sizes ** 2) - indicator[i]) / n
        elif score_rule == 'H':
            probs = partition_sizes / n
            entropy = -np.sum(probs * np.log2(probs))
            scores[i] = -entropy
        else:
            raise ValueError(f"Unknown score rule: {score_rule}")

    sorted_indices = np.argsort(scores, kind='stable')
    g_star_idxs = sorted_indices[:num_of_guesses]
    picks = G[g_star_idxs]
    indicator_picks = [bool(indicator[idx]) for idx in g_star_idxs]
    return picks, indicator_picks


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10, score_rule='PC', configs=None):
    """
    Finds the top-k best guesses by minimizing the specified score rule (GPU).
    """
    base = configs.get('base', 243)
    n, nG = len(T), len(G)
    guesses_include_targets = bool(configs.get('guesses_include_targets', True))

    feedbacks_sub = F[T[:, None], G]
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    # Pre-compute indicator
    if guesses_include_targets:
        indicator = cp.isin(G, T)
    else:
        indicator = cp.zeros(nG, dtype=bool)

    if score_rule == 'PC':
        num_feedbacks = (counts > 0).sum(axis=1)
        scores = (n - indicator) / num_feedbacks
    elif score_rule == 'WA':
        scores = ((counts ** 2).sum(axis=1) - indicator) / n
    elif score_rule == 'H':
        mask = counts > 0
        probs = counts / n
        log_probs = cp.where(mask, cp.log2(probs), 0.0)
        entropy = -cp.sum(probs * log_probs, axis=1)
        scores = -entropy
    else:
        raise ValueError(f"Unknown score rule: {score_rule}")

    sorted_indices = cp.argsort(scores)
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs], indicator[g_star_idxs]


def _get_best_guess_subtree(T, G, F, subtree, configs):
    """
    Finds the best guess using a subtree metric.
    """
    n = len(T)
    guesses_include_targets = bool(configs.get('guesses_include_targets', True))

    # Shortcut for small target sets
    shortcut_threshold = 2 if guesses_include_targets else 1
    if n <= shortcut_threshold:
        return T[0], guesses_include_targets

    T, G, xp, F, _, _, get_best_guesses = subtree.optimizer.get_context(T, G)

    if configs['metric'] == 1:
        # Subtree-k: evaluate top-k candidates
        G_prime, candidates_in_T = get_best_guesses(T, G, F)
    else:
        # Subtree-full: evaluate all guesses
        if guesses_include_targets:
            candidates_in_T = xp.isin(G, T).tolist()
        else:
            candidates_in_T = [False] * len(G)
        G_prime = G

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
