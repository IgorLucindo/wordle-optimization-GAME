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

    def _get_best_guess_CPU(T, G, F):
        return _get_best_guess_CPU_impl(
            T, G, F, base=base, targets_have_self_id=targets_have_self_id)

    def _get_best_guess_GPU(T, G, F):
        return _get_best_guess_GPU_impl(
            T, G, F, base=base, targets_have_self_id=targets_have_self_id)

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

    def get_best_guesses_CPU(T, G, F):
        return _get_best_guesses_CPU(T, G, F, num_of_guesses=configs['k'],
                                      base=base,
                                      targets_have_self_id=targets_have_self_id)

    def get_best_guesses_GPU(T, G, F):
        return _get_best_guesses_GPU(T, G, F, num_of_guesses=configs['k'],
                                      base=base,
                                      targets_have_self_id=targets_have_self_id)

    return (get_best_guesses_CPU, get_best_guesses_GPU)


def _get_best_guess_CPU_impl(T, G, F, base=243, targets_have_self_id=True):
    """
    Finds the best guess by minimizing the average size of remaining set (CPU).

    ``base`` is the feedback-code upper bound (bincount minlength).
    ``targets_have_self_id`` is False for games where targets are not valid
    guesses (e.g. Zoo with attribute-only queries); in that case the
    ``n <= 2`` shortcut is disabled, the (n - 1) self-id credit in the score
    is suppressed, and ``g_star_in_T`` is always False -- a numeric
    coincidence between a guess index (attribute column) and a target index
    (animal row) carries no semantic meaning.
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
        P_g = np.nonzero(np.bincount(F[T, g], minlength=base))[0]
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    argmin = int(np.argmin(scores))
    g_star = G[argmin]
    g_star_in_T = (int(g_star) in T_set) if targets_have_self_id else False
    return g_star, g_star_in_T


def _get_best_guess_GPU_impl(T, G, F, base=243, targets_have_self_id=True):
    """
    Finds the best guess by minimizing the average size of remaining set (GPU).

    See ``_get_best_guess_CPU_impl`` for the semantics of
    ``targets_have_self_id``.
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
    num_feedbacks = (counts > 0).sum(axis=1)

    # Score. Without self-id (Zoo), the ``-in_T`` correction is meaningless
    # because the overlap between guess-column indices and target-row indices
    # is coincidental.
    if targets_have_self_id:
        global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
        global_mask[T] = True
        in_T = global_mask[G]
        scores = (n - in_T) / num_feedbacks
    else:
        in_T = cp.zeros(nG, dtype=cp.bool_)
        scores = n / num_feedbacks

    # Best guess
    argmin = cp.argmin(scores)
    g_star = G[argmin]
    return g_star, in_T[argmin]


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10, base=243,
                          targets_have_self_id=True):
    """
    Finds the best guesses by minimizing the average size of remaining set (CPU).
    """
    n = len(T)
    scores = np.empty(len(G), dtype=np.float64)
    if targets_have_self_id:
        T_set = set(T.tolist())
    else:
        T_set = set()

    for i, g in enumerate(G):
        g = int(g)
        P_g = np.nonzero(np.bincount(F[T, g], minlength=base))[0]
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    sorted_indices = np.argsort(scores, kind='stable')
    g_star_idxs = sorted_indices[:num_of_guesses]
    picks = G[g_star_idxs]
    if targets_have_self_id:
        in_T = [int(g) in T_set for g in picks]
    else:
        in_T = [False] * len(picks)
    return picks, in_T


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10, base=243,
                          targets_have_self_id=True):
    """
    Finds the best guesses by minimizing the average size of remaining set (GPU).
    """
    n, nG = len(T), len(G)

    feedbacks_sub = F[T[:, None], G]
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)
    num_feedbacks = (counts > 0).sum(axis=1)

    if targets_have_self_id:
        global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
        global_mask[T] = True
        in_T = global_mask[G]
        scores = (n - in_T) / num_feedbacks
    else:
        in_T = cp.zeros(nG, dtype=cp.bool_)
        scores = n / num_feedbacks

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
            G_prime, candidates_in_T = G, global_mask[G]
        else:
            G_prime = G
            candidates_in_T = xp.zeros(len(G), dtype=xp.bool_)

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
    return _get_best_guess_CPU_impl(T, G, F, base=243)


def _get_best_guess_GPU(T, G, F):
    return _get_best_guess_GPU_impl(T, G, F, base=243)
