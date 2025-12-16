from classes.guess_tree import *
from cupyx import scatter_add
import numpy as np
import cupy as cp


def best_guess_function(instance_data, flags, configs):
    """
    Selects the appropriate best guess function based on flags
    """
    get_best_guess = _get_best_guess_GPU if configs['GPU'] else _get_best_guess_CPU
    get_best_guesses = best_guesses_function(configs)
    instance_for_subtree = instance_data + (get_best_guess, get_best_guesses,)
    
    # Get best guess function if subtree metric is choosen
    if configs['subtree_score']:
        subtree = Guess_Tree(instance_for_subtree, flags, configs)
        
        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree)
        get_best_guess = get_best_guess_subtree

    return get_best_guess


def best_guesses_function(configs):
    return _get_best_guesses_GPU if configs['GPU'] else _get_best_guesses_CPU


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by minimizing the expected size of remaining set (CPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]
    
    scores = np.zeros(len(G))
    T_set = set(T.tolist())

    for i, g in enumerate(G):
        P_g = np.unique(F[T, g])
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    # Best guess
    g_star = G[np.argmin(scores)]
    return g_star


def _get_best_guess_GPU(T, G, F):
    """
    Finds the best guess by minimizing the expected size of remaining set (GPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]

    nG = len(G)
    base = 243  # Number of possible feedback patterns

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]

    # Flatten in column-major order to align with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # Build histogram table [guess × feedback]
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))
    num_feedbacks = (hist > 0).sum(axis=1)
    global_mask = cp.zeros(F.shape[0], dtype=cp.bool_)
    global_mask[T] = True
    in_T = global_mask[G]

    # Score
    scores = (n - in_T) / num_feedbacks

    # Best guess
    g_star = G[cp.argmin(scores)]
    return g_star


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10):
    """
    Finds the best guesses by minimizing the expected size of remaining set (CPU)
    """
    n = len(T)
    scores = np.zeros(len(G))
    T_set = set(T.tolist())

    for i, g in enumerate(G):
        P_g = np.unique(F[T, g])
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    # Best guesses
    sorted_indices = np.argsort(scores, kind='stable')
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs]


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10):
    """
    Finds the best guesses by minimizing the expected size of remaining set (GPU)
    """
    n = len(T)
    nG = len(G)
    base = 243  # Number of possible feedback patterns

    # Extract feedback submatrix for feasible targets vs all guesses
    feedbacks_sub = F[T[:, None], G]

    # Flatten in column-major order to align correctly with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # Build histogram table [guess × feedback]
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))
    num_feedbacks = (hist > 0).sum(axis=1)
    global_mask = cp.zeros(F.shape[0], dtype=cp.bool_)
    global_mask[T] = True
    in_T = global_mask[G]

    # Score
    scores = (n - in_T) / num_feedbacks

    # Best guesses
    sorted_indices = cp.argsort(scores)
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs]


def _get_best_guess_subtree(T, G, F, subtree):
    """
    Finds the best guess using a subtree metric
    """
    n = len(T)
    if n <= 2:
        return T[0]
    
    xp = subtree.xp
    
    # Get guess candidates based on chosen metric
    g_candidates = subtree.get_best_guesses(T, G, F)
    # g_candidates = G
    
    scores = xp.zeros(len(g_candidates))
    subtree.T = T
    subtree.G = G

    for i, g in enumerate(g_candidates):
        subtree.starting_word = g
        subtree.build(build_flag=False)
        subtree.evaluate_quick()
        scores[i] = subtree.results['exp_guesses'] + 0.001 * subtree.results['max_guesses']

    # Best guess
    g_star = g_candidates[xp.argmin(scores)]
    return g_star