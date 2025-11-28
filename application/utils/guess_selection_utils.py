from classes.guess_tree import *
from cupyx import scatter_add
import numpy as np
import cupy as cp


def best_guess_function(instance_data, flags, configs):
    """
    Selects the appropriate best guess function based on flags
    """
    get_best_guess = _get_best_guess_GPU if configs['GPU'] else _get_best_guess_CPU
    instance_for_subtree = instance_data + (get_best_guess,)
    
    # Get best guess function if subtree metric is choosen
    if configs['subtree_score']:
        subtree = Guess_Tree(instance_for_subtree, flags, configs)
        
        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree)
        get_best_guess = get_best_guess_subtree

    return get_best_guess


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by minimizing the expected size of remaining set (CPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]
    
    scores = np.zeros(len(G))
    T_set = set(T.tolist())

    for g in G:
        P_g = np.unique(F[T, g])
        scores[g] = ((n - 1) if g in T_set else n) / len(P_g)

    # Get best guess
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
    feedbacks_sub = F[T[:, None], G]  # shape: (len(T), len(G))

    # Flatten in column-major order to align with guesses
    flat_fb = feedbacks_sub.ravel(order='F')
    flat_col = cp.repeat(cp.arange(nG), n)

    # Build histogram table [guess × feedback]
    hist = cp.zeros((nG, base), dtype=cp.int32)
    scatter_add(hist, (flat_col, flat_fb), cp.ones_like(flat_fb, dtype=cp.int32))
    num_feedbacks = (hist > 0).sum(axis=1)
    hist[:, 242] = 0

    # Compute per-guess statistics
    sum_counts = hist.sum(axis=1)
    mean_counts = sum_counts / num_feedbacks

    # Score
    scores = mean_counts

    # Best guess
    g_star = G[cp.argmin(scores)]
    return g_star


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10):
    """
    Finds the best guesses by minimizing the expected size of remaining set (GPU)
    """
    n = len(T)
    if n <= 2:
        return T[0]

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
    hist[:, 242] = 0

    # Compute per-guess statistics
    sum_counts = hist.sum(axis=1)
    mean_counts = sum_counts / num_feedbacks

    # Score
    scores = mean_counts

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
    
    # Get guess candidates based on chosen metric
    g_candidates = _get_best_guesses_GPU(T, G, F)
    
    min_score = float('inf')
    subtree.T = T
    subtree.G = G

    for g in g_candidates:
        subtree.starting_word = g
        subtree.build(build_flag=False)

        score = subtree.results['avg_guesses']
        if score < min_score:
            min_score = score
            g_star = g

    return g_star