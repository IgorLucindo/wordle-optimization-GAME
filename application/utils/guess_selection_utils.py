from classes.guess_tree import *
import numpy as np
import cupy as cp


def best_guess_functions(instance_data, flags, configs):
    """
    Selects the appropriate best guess function based on flags
    """
    _best_guess_functions = (_get_best_guess_CPU, _get_best_guess_GPU)
    _best_guesses_functions = best_guesses_functions()
    
    # Get best guess function if subtree metric is choosen
    if configs['metric']:
        instance = instance_data + (_best_guess_functions, _best_guesses_functions)
        subtree = Guess_Tree(instance, flags, configs)
        
        def get_best_guess_subtree(T, G, F):
            return _get_best_guess_subtree(T, G, F, subtree, configs)
        _best_guess_functions = (get_best_guess_subtree, get_best_guess_subtree)

    return _best_guess_functions


def best_guesses_functions():
    return _get_best_guesses_CPU, _get_best_guesses_GPU


def _get_best_guess_CPU(T, G, F):
    """
    Finds the best guess by minimizing the average size of remaining set (CPU)
    """
    n = len(T)
    if n <= 2:
        return T[0], True
    
    scores = np.zeros(len(G))
    T_set = set(T.tolist())

    for i, g in enumerate(G):
        P_g = np.nonzero(np.bincount(F[T, g], minlength=243))[0]
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    # Best guess
    argmin = np.argmin(scores)
    g_star = G[argmin]
    return g_star, g_star in T_set


def _get_best_guess_GPU(T, G, F):
    """
    Finds the best guess by minimizing the average size of remaining set (GPU)
    """
    n = len(T)
    if n <= 2:
        return T[0], True
    
    nG = len(G)
    base = 243

    # Extract feedback submatrix
    feedbacks_sub = F[T[:, None], G]

    # Linear Indexing
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    # Bincount
    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)
    num_feedbacks = (counts > 0).sum(axis=1)

    # Score
    global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
    global_mask[T] = True
    in_T = global_mask[G]

    scores = (n - in_T) / num_feedbacks

    # Best guess
    argmin = cp.argmin(scores)
    g_star = G[argmin]
    return g_star, in_T[argmin]


def _get_best_guesses_CPU(T, G, F, num_of_guesses=10):
    """
    Finds the best guesses by minimizing the average size of remaining set (CPU)
    """
    n = len(T)
    scores = np.zeros(len(G))
    T_set = set(T.tolist())

    for i, g in enumerate(G):
        P_g = np.nonzero(np.bincount(F[T, g], minlength=243))[0]
        scores[i] = ((n - 1) if g in T_set else n) / len(P_g)

    # Best guesses
    sorted_indices = np.argsort(scores, kind='stable')
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs], [g in T_set for g in G[g_star_idxs]]


def _get_best_guesses_GPU(T, G, F, num_of_guesses=10):
    """
    Finds the best guesses by minimizing the average size of remaining set (GPU)
    """
    n, nG = len(T), len(G)
    base = 243

    # Extract feedback submatrix for feasible targets vs all guesses
    feedbacks_sub = F[T[:, None], G]

    # Linear Indexing
    offsets = cp.arange(nG, dtype=cp.int32) * base
    global_indices = feedbacks_sub + offsets

    # Bincount
    counts_flat = cp.bincount(global_indices.ravel(), minlength=nG * base)
    counts = counts_flat.reshape(nG, base)
    num_feedbacks = (counts > 0).sum(axis=1)

    # Score
    global_mask = cp.zeros(F.shape[1], dtype=cp.bool_)
    global_mask[T] = True
    in_T = global_mask[G]

    scores = (n - in_T) / num_feedbacks

    # Best guesses
    sorted_indices = cp.argsort(scores)
    g_star_idxs = sorted_indices[:num_of_guesses]
    return G[g_star_idxs], in_T[g_star_idxs]


def _get_best_guess_subtree(T, G, F, subtree, configs):
    """
    Finds the best guess using a subtree metric
    """
    n = len(T)
    if n <= 2:
        return T[0], True
    
    # Ask Optimizer for Context
    T, G, xp, F, _, _, get_best_guesses = subtree.optimizer.get_context(T, G)
    
    # Get guess candidates based on chosen metric
    if configs['metric'] == 1:
        G_prime, candidates_in_T = get_best_guesses(T, G, F)
    else:
        global_mask = xp.zeros(F.shape[1], dtype=xp.bool_)
        global_mask[T] = True
        G_prime, candidates_in_T = G, global_mask[G]
    
    scores = np.zeros(len(G_prime))
    subtree.T = T
    subtree.G = G

    for i, g_start in enumerate(G_prime):
        g_start_in_T = candidates_in_T[i]
        D = subtree.build_subtree(g_start, g_start_in_T)
        scores[i] = D.mean() + 0.001 * D.max()

    # Best guess
    argmin = np.argmin(scores)
    g_star = G_prime[argmin]
    return g_star, candidates_in_T[argmin]