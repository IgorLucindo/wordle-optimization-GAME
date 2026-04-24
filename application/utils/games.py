"""
Game abstraction layer for the generalized guessing-game solver.

Each game exposes a loader that returns a dict with keys:

    G_names         : list[str]          display names for guesses,  len = n_G
    T_names         : list[str]          display names for targets,  len = n_T
    F               : np.ndarray         shape (n_T, n_G), integer feedback codes
    base            : int                upper bound on feedback codes (bincount minlength)
    is_target       : np.ndarray(bool)   shape (n_G,) -- True iff guess g also
                                          acts as a terminal identification action
                                          for some target
    decode_feedback : callable(code)->Any  readable feedback (tuple/dict) for saved trees
    hard_mode_supported : bool

For the three instances used in the paper:

* wordle     n_T=2315, n_G=12972, base=243, ternary feedback codes (5 positions x 3 colors)
* mastermind n_T=n_G=1296 (4 pegs, 6 colors), base=(pegs+1)^2=25, black/white feedback
* zoo        n_T=59 (feature-distinct animal classes), n_G=n_T+16 (self-id actions
             prepended to attribute actions), base=9 (max attribute value + 1),
             feedback = attribute value on target
"""
from pathlib import Path
import itertools
import csv
import numpy as np


# ---------------------------------------------------------------------------
# Wordle
# ---------------------------------------------------------------------------

def _wordle_feedback_matrix_cpu(T, G):
    """Wordle feedback matrix (CPU). Encodes the 5-position ternary code
    (green=2, yellow=1, gray=0) as a single base-3 integer in [0, 243)."""
    K, nG, L = len(T), len(G), 5
    T_int = [[ord(c) - 97 for c in w] for w in T]
    G_int = [[ord(c) - 97 for c in w] for w in G]
    F = np.zeros((K, nG), dtype=np.uint8)
    powers = [3 ** (L - 1 - i) for i in range(L)]
    c_t = np.zeros((K, 26), dtype=np.int16)
    for t_idx, t_word in enumerate(T_int):
        for ch in t_word:
            c_t[t_idx][ch] += 1
    for t_idx in range(K):
        target = T_int[t_idx]
        for g_idx in range(nG):
            guess = G_int[g_idx]
            f_row = [0] * L
            c_prime = c_t[t_idx].copy()
            for i in range(L):
                if guess[i] == target[i]:
                    f_row[i] = 2
                    c_prime[guess[i]] -= 1
            for i in range(L):
                if f_row[i] == 0 and c_prime[guess[i]] > 0:
                    f_row[i] = 1
                    c_prime[guess[i]] -= 1
            code = 0
            for i in range(L):
                code += f_row[i] * powers[i]
            F[t_idx, g_idx] = code
    return F


def _wordle_decode_feedback(code):
    base, L = 3, 5
    out = []
    c = int(code)
    for i in range(L):
        out.append((c // (base ** (L - 1 - i))) % base)
    return tuple(out)


def load_wordle(dataset_dir='data'):
    """Load Wordle. We return the word-level instance and let the caller build F
    via the original instance_utils.py path (which supports GPU/hard-mode variants).
    The F returned here is only used by the generic loader; the production
    Wordle path still goes through ``_get_feedback_matrix`` in instance_utils.py.
    """
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / 'solutions.txt') as f:
        T = [line.strip() for line in f if line.strip()]
    with open(dataset_dir / 'non_solutions.txt') as f:
        non = [line.strip() for line in f if line.strip()]
    G = T + non
    n_T, n_G = len(T), len(G)
    F = _wordle_feedback_matrix_cpu(T, G)
    is_target = np.zeros(n_G, dtype=bool)
    is_target[:n_T] = True
    return {
        'G_names': G, 'T_names': T,
        'F': F, 'base': 243,
        'is_target': is_target,
        'targets_have_self_id': True,
        'decode_feedback': _wordle_decode_feedback,
        'hard_mode_supported': True,
    }


# ---------------------------------------------------------------------------
# Mastermind (pegs x colors, codes with repetition)
# ---------------------------------------------------------------------------

def load_mastermind(dataset_dir='data', pegs=4, colors=6):
    codes = list(itertools.product(range(colors), repeat=pegs))
    n = len(codes)
    names = ["".join(str(x) for x in c) for c in codes]
    codes_arr = np.array(codes, dtype=np.int8)

    # Black counts: F_black[i_t, i_g] = # positions where codes agree
    black = (codes_arr[:, None, :] == codes_arr[None, :, :]).sum(axis=2).astype(np.int16)

    # Per-code color counts, shape (n, colors)
    counts = np.zeros((n, colors), dtype=np.int16)
    for c in range(colors):
        counts[:, c] = (codes_arr == c).sum(axis=1)

    # Common multiset intersection, shape (n, n)
    common = np.minimum(counts[:, None, :], counts[None, :, :]).sum(axis=2).astype(np.int16)
    white = common - black

    pegs_plus_1 = pegs + 1
    F = (black * pegs_plus_1 + white).astype(np.uint8)  # max value (pegs+1)*pegs = 20 for 4x6
    base = pegs_plus_1 * pegs_plus_1  # 25 for 4x6

    is_target = np.ones(n, dtype=bool)

    def decode_feedback(code):
        code = int(code)
        return (code // pegs_plus_1, code % pegs_plus_1)

    return {
        'G_names': names, 'T_names': names[:],
        'F': F, 'base': base,
        'is_target': is_target,
        'targets_have_self_id': True,
        'decode_feedback': decode_feedback,
        'hard_mode_supported': False,
    }


# ---------------------------------------------------------------------------
# UCI Zoo (binary attributes + legs with values in {0,2,4,5,6,8})
# ---------------------------------------------------------------------------

def _read_zoo_csv(path):
    animals, features = [], []
    with open(path, newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            animals.append(row[0])
            features.append(tuple(int(x) for x in row[1:]))
    return header[1:], animals, features


def load_zoo(dataset_dir='data'):
    """Zoo instance.

    Targets are feature-distinct animal classes (deduplicated). Guesses are
    the 16 attributes only -- there is no "self-identification" action in
    classical sequential testing. Consequently ``is_target`` is all False;
    the Guess_Tree builder detects the |T'|==1 leaf case explicitly and
    stops recursion without recording a terminal guess. Depth of a leaf
    equals the number of attribute queries on the root-to-leaf path.

    The solver's ``|T'| <= 2`` shortcut is bypassed when the shortcut would
    return a target-index that is not a valid guess (see
    ``_get_best_guess_CPU_impl`` which checks ``is_target[T[0]]``).
    """
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / 'zoo.csv'
    attr_names, animals, features = _read_zoo_csv(path)

    # Deduplicate: animals with identical feature vectors are merged.
    seen, uniq_animals, uniq_feats = {}, [], []
    for a, f in zip(animals, features):
        if f in seen:
            continue
        seen[f] = a
        uniq_animals.append(a)
        uniq_feats.append(f)
    n_T = len(uniq_animals)
    feats_arr = np.array(uniq_feats, dtype=np.int16)  # (n_T, n_attrs)
    n_attrs = feats_arr.shape[1]

    F = feats_arr.astype(np.uint8)   # shape (n_T, n_attrs); F[t, a] = value
    max_val = int(feats_arr.max())
    base = max(2, max_val + 1)

    G_names = list(attr_names)
    T_names = list(uniq_animals)

    is_target = np.zeros(n_attrs, dtype=bool)   # attributes are not targets

    def decode_feedback(code):
        return (int(code),)

    return {
        'G_names': G_names, 'T_names': T_names,
        'F': F, 'base': base,
        'is_target': is_target,
        'targets_have_self_id': False,
        'decode_feedback': decode_feedback,
        'hard_mode_supported': False,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    'wordle': load_wordle,
    'mastermind': load_mastermind,
    'zoo': load_zoo,
}


def load_game(game_name, dataset_dir='data'):
    if game_name not in LOADERS:
        raise ValueError(f"Unknown game '{game_name}'. Supported: {list(LOADERS)}")
    return LOADERS[game_name](dataset_dir=dataset_dir)
