from classes.guess_tree import *
from utils.xp_utils import cp, HAS_CUPY
import numpy as np


def best_guess_functions(instance_data, flags, configs, base=243):
    """
    Selects the appropriate best guess functions.

    ``base`` is the per-game upper bound on feedback codes (bincount minlength);
    see utils.games for per-game values (243 for Wordle, 25 for Mastermind 4x6,
    9 for UCI Zoo).
    """
    targets_have_self_id = bool(configs.get('targets_have_self_id', True))
    score_rule = configs.get('score', 'PC')

    def _get_best_guess_CPU(T, G, F):
        return _get_best_guess_CPU_impl(
            T, G, F, base=base, targets_have_self_id=targets_have_self_id,
            score_rule=score_rule)

    def _get_best_guess_GPU(T, G, F):
        return _get_best_guess_GPU_impl(
            T, G, F, base=base, targets_have_self_id=targets_have_self_id,
            score_rule=score_rule)

    _best_guess_functions = (_get_best_guess_CPU, _get_best_guess_GPU)
    _best_guesses_functions = best_guesses_functions(configs, base=base)

    # Subtree look-ahead wraps the avg-size rule.
    if configs['metric']:
        instance = instance_data + (_best_guess_functions, _best_guesses_functions)
        subtree = Guess_Tree(instance, flags, configs)

        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree, configs)
        _best_guess_functions = (get_best_guess_subtree, get_best_guess_subtree)

    return _best_guess_functions


def best_guesses_functions(configs, base=243):
    """Top-k candidate generators for Subtree-k."""
    targets_have_self_id = bool(configs.get('targets_have_self_id', True))
    score_rule = configs.get('score', 'PC')

    def get_best_guesses_CPU(T, G, F):
        return _get_best_guesses_CPU(T, G, F, num_of_guesses=configs['k'],
                                      base=base,
                                      targets_have_self_id=targets_have_self_id,
                                      score_rule=score_rule)

    def get_best_guesses_GPU(T, G, F):
        return _get_best_guesses_GPU(T, G, F, num_of_guesses=configs['k'],
                                      base=base,
                                      targets_have_self_id=targets_have_self_id,
                                      score_rule=score_rule)

    return (get_best_guesses_CPU, get_best_guesses_GPU)


def _get_best_guess_CPU_impl(T, G, F, base=243, targets_have_self_id=True, score_rule='PC'):
    """
    Finds the best guess by minimizing the specified score rule (CPU).

    ``base`` is the feedback-code upper bound (bincount minlength).
    ``targets_have_self_id`` is False for games where targets are not valid
    guesses (e.g. Zoo with attribute-only queries); in that case the
    ``n <= 2`` shortcut is disabled, the (n - 1) self-id credit in the score
    is suppressed, and ``g_star_in_T`` is always False -- a numeric
    coincidence between a guess index (attribute column) and a target index
    (animal row) carries no semantic meaning.
    ``score_rule`` specifies the scoring rule: 'PC' (Partition Count),
    'WA' (Weighted Average), or 'H' (Entropy).
    """
    n = len(T)
    if n <= 2 and targets_have_self_id:
        return T[0], True

    scores = np.empty(len(G), dtype=np.float64)
    if targets_have_self_id:
        T_set = set(T.tolist())
    else:
        T_set = set()

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]  # Keep only non-empty partitions
        indicator = 1 if (g in T_set) else 0

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
    g_star_in_T = (int(g_star) in T_set) if targets_have_self_id else False
    return g_star, g_star_in_T


def _get_best_guess_GPU_impl(T, G, F, base=243, targets_have_self_id=True, score_rule='PC'):
    """
    Finds the best guess by minimizing the specified score rule (GPU).

    See ``_get_best_guess_CPU_impl`` for the semantics of
    ``targets_have_self_id`` and ``score_rule``.
    """
    n = len(T)
    if n <= 2 and targets_have_self_id:
        return T[0], True

    nG = len(G)

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]

    # Linear Indexing
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    # Bincount over the flattened (target, guess) entries
    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    # Determine indicator function
    if targets_have_self_id:
        global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
        global_mask[T] = True
        indicator = global_mask[G]
    else:
        indicator = cp.zeros(nG, dtype=cp.bool_)

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


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10, base=243,
                          targets_have_self_id=True, score_rule='PC'):
    """
    Finds the best guesses by minimizing the specified score rule (CPU).
    """
    n = len(T)
    scores = np.empty(len(G), dtype=np.float64)
    if targets_have_self_id:
        T_set = set(T.tolist())
    else:
        T_set = set()

    for i, g in enumerate(G):
        g = int(g)
        partition_sizes = np.bincount(F[T, g], minlength=base)
        partition_sizes = partition_sizes[partition_sizes > 0]

        n_adjusted = (n - 1) if g in T_set else n

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
    if targets_have_self_id:
        in_T = [int(g) in T_set for g in picks]
    else:
        in_T = [False] * len(picks)
    return picks, in_T


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10, base=243,
                          targets_have_self_id=True, score_rule='PC'):
    """
    Finds the best guesses by minimizing the specified score rule (GPU).
    """
    n, nG = len(T), len(G)

    feedbacks_sub = F[T[:, None], G]
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)

    if targets_have_self_id:
        global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
        global_mask[T] = True
        in_T = global_mask[G]
        n_adjusted = n - in_T
    else:
        in_T = cp.zeros(nG, dtype=cp.bool_)
        n_adjusted = cp.full(nG, n, dtype=cp.float64)

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


def _get_best_guess_subtree(T, G, F, subtree, configs):
    """
    Finds the best guess using a subtree metric
    """
    targets_have_self_id = bool(configs.get('targets_have_self_id', True))

    n = len(T)
    if n <= 2 and targets_have_self_id:
        return T[0], True

    T, G, xp, F, _, _, get_best_guesses = subtree.optimizer.get_context(T, G)

    if configs['metric'] == 1:
        G_prime, candidates_in_T = get_best_guesses(T, G, F)
    else:
        if targets_have_self_id:
            global_mask = xp.zeros(F.shape[1], dtype=xp.bool_)
            global_mask[T] = True
            G_prime, candidates_in_T = G, global_mask[G].tolist()
        else:
            G_prime = G
            candidates_in_T = [False] * len(G)

    g_star, g_star_in_T = _get_best_subtree_candidate(T, G, G_prime, F, candidates_in_T, subtree)
    return g_star, g_star_in_T


def _get_best_subtree_candidate(T, G, G_prime, F, candidates_in_T, subtree):
    """
    Evaluates the provided candidates (G_prime) using the subtree metric and selects the best one
    """
    subtree.T = T
    subtree.G = G
    scores = np.zeros(len(G_prime))

    for i, g_start in enumerate(G_prime):
        g_start_in_T = candidates_in_T[i]
        D = subtree.build_subtree(g_start, g_start_in_T)
        scores[i] = D.mean() + 0.001 * D.max()

    argmin = np.argmin(scores)
    g_star = G_prime[argmin]
    return g_star, candidates_in_T[argmin]


# ---------------------------------------------------------------------------
# Back-compat shims: original Wordle code used `_get_best_guess_CPU` with no
# ``base`` argument. Keep the legacy symbol aliased to the base=243 path so
# any caller that imports `_get_best_guess_CPU` by name still works.
# ---------------------------------------------------------------------------
def _get_best_guess_CPU(T, G, F):
    return _get_best_guess_CPU_impl(T, G, F, base=243, score_rule='PC')


def _get_best_guess_GPU(T, G, F):
    return _get_best_guess_GPU_impl(T, G, F, base=243, score_rule='PC')
