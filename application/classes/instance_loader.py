"""
Unified instance loader for the decision tree solver.

Encapsulates all instance loading, format handling, feedback matrix building,
GPU conversion, and setup logic in a single class.
"""
from pathlib import Path
from utils.xp_utils import cp, HAS_CUPY
from utils.guess_selection_utils import best_guess_functions, best_guesses_functions
import itertools
import csv
import json
import numpy as np


class InstanceLoader:
    """
    Unified instance loader that encapsulates all instance loading and setup.

    Loads instance configuration from data/<instance_name>/rules.json,
    builds feedback matrices, converts data to GPU if requested, and provides
    best guess functions.
    """

    def __init__(self, instance_name, use_gpu=False, dataset_dir='data'):
        """
        Load and setup an instance.

        Args:
            instance_name: Name of instance (e.g., 'wordle', 'mastermind', 'zoo')
            use_gpu: Whether to use GPU acceleration (requires CuPy)
            dataset_dir: Root directory containing instance folders
        """
        self.instance_name = instance_name
        self.use_gpu = use_gpu and HAS_CUPY

        # Force CPU for non-Wordle games (no GPU implementation yet)
        if instance_name not in ('wordle', 'wordle_hard') and self.use_gpu:
            self.use_gpu = False

        # Load raw data from disk
        data = self._load_from_disk(dataset_dir)

        # Extract and store instance properties
        self.G_names = data['G_names']
        self.T_names = data['T_names']
        self.decode_feedback = data['decode_feedback']

        # Rules from config file
        rules = data['rules']
        self.base = rules['base']
        self.guesses_include_targets = rules.get('guesses_include_targets', True)
        self.constrained_guessing = rules.get('constrained_guessing', False)

        # Convert is_target to GPU if needed
        self.F = data['F']
        self.is_target = cp.array(data['is_target']) if self.use_gpu else data['is_target']

        # Build compatibility matrix for constrained guessing
        self.C = self._build_compatibility_matrix() if self.constrained_guessing else None

    def _load_from_disk(self, dataset_dir):
        """
        Load instance from disk based on rules.json configuration.

        Returns dict with standardized keys:
            G_names, T_names, F, base, is_target, decode_feedback, rules
        """
        instance_path = Path(dataset_dir) / self.instance_name
        rules_path = instance_path / 'rules.json'

        if not rules_path.exists():
            raise FileNotFoundError(f"Configuration not found: {rules_path}")

        with open(rules_path) as f:
            rules = json.load(f)

        # Dispatch by format
        format_type = rules.get('format')
        if format_type == 'wordlist':
            return self._load_wordlist_format(instance_path, rules)
        elif format_type == 'csv':
            return self._load_csv_format(instance_path, rules)
        elif format_type == 'generated':
            return self._load_generated_format(rules)
        else:
            raise ValueError(f"Unknown format '{format_type}' in {rules_path}")

    def _load_wordlist_format(self, instance_path, rules):
        """Load wordlist-based instances (Wordle)."""
        data_files = rules['data_files']
        targets_file = instance_path / data_files['targets']
        guesses_file = instance_path / data_files.get('guesses', data_files['targets'])

        with open(targets_file) as f:
            T = [line.strip() for line in f if line.strip()]

        if guesses_file.exists():
            with open(guesses_file) as f:
                non = [line.strip() for line in f if line.strip()]
            G = T + non
        else:
            G = T

        n_T, n_G = len(T), len(G)

        # For constrained guessing mode, we need full (n_G, n_G) feedback matrix
        # For normal mode, we only need (n_T, n_G)
        if rules.get('constrained_guessing', False):
            F = self._build_wordle_feedback_matrix(G, G, use_gpu=self.use_gpu)
        else:
            F = self._build_wordle_feedback_matrix(T, G, use_gpu=self.use_gpu)

        # First n_T guesses are targets
        is_target = np.zeros(n_G, dtype=bool)
        is_target[:n_T] = True

        return {
            'G_names': G,
            'T_names': T,
            'F': F,
            'base': rules['base'],
            'is_target': is_target,
            'decode_feedback': self._wordle_decode_feedback,
            'rules': rules
        }

    def _load_csv_format(self, instance_path, rules):
        """Load CSV-based instances (Zoo)."""
        data_files = rules['data_files']
        csv_path = instance_path / data_files['dataset']

        attr_names, animals, features = self._read_csv(csv_path)

        # Deduplicate: animals with identical feature vectors are merged
        seen, uniq_animals, uniq_feats = {}, [], []
        for a, f in zip(animals, features):
            if f in seen:
                continue
            seen[f] = a
            uniq_animals.append(a)
            uniq_feats.append(f)

        n_T = len(uniq_animals)
        feats_arr = np.array(uniq_feats, dtype=np.int16)
        n_attrs = feats_arr.shape[1]

        F = feats_arr.astype(np.uint8)
        max_val = int(feats_arr.max())
        base = max(2, max_val + 1)

        # Attributes are NOT targets (no self-identification)
        is_target = np.zeros(n_attrs, dtype=bool)

        def decode_feedback(code):
            return (int(code),)

        return {
            'G_names': list(attr_names),
            'T_names': list(uniq_animals),
            'F': F,
            'base': base,
            'is_target': is_target,
            'decode_feedback': decode_feedback,
            'rules': rules
        }

    def _load_generated_format(self, rules):
        """Load programmatically generated instances (Mastermind)."""
        params = rules['generator_params']
        pegs = params['pegs']
        colors = params['colors']

        codes = list(itertools.product(range(colors), repeat=pegs))
        n = len(codes)
        names = ["".join(str(x) for x in c) for c in codes]
        codes_arr = np.array(codes, dtype=np.int8)

        # Black counts: F_black[i_t, i_g] = # positions where codes agree
        black = (codes_arr[:, None, :] == codes_arr[None, :, :]).sum(axis=2).astype(np.int16)

        # Per-code color counts
        counts = np.zeros((n, colors), dtype=np.int16)
        for c in range(colors):
            counts[:, c] = (codes_arr == c).sum(axis=1)

        # Common multiset intersection
        common = np.minimum(counts[:, None, :], counts[None, :, :]).sum(axis=2).astype(np.int16)
        white = common - black

        pegs_plus_1 = pegs + 1
        F = (black * pegs_plus_1 + white).astype(np.uint8)
        base = pegs_plus_1 * pegs_plus_1

        # All codes can be targets (self-identifying)
        is_target = np.ones(n, dtype=bool)

        def decode_feedback(code):
            code = int(code)
            return (code // pegs_plus_1, code % pegs_plus_1)

        return {
            'G_names': names,
            'T_names': names[:],
            'F': F,
            'base': base,
            'is_target': is_target,
            'decode_feedback': decode_feedback,
            'rules': rules
        }

    def _build_wordle_feedback_matrix(self, T, G, use_gpu=False):
        """
        Build Wordle feedback matrix on CPU or GPU.

        Encodes the 5-position ternary code (green=2, yellow=1, gray=0)
        as a single base-3 integer in [0, 243).
        """
        if use_gpu:
            return self._build_wordle_feedback_matrix_GPU(T, G)
        else:
            return self._build_wordle_feedback_matrix_CPU(T, G)

    def _build_wordle_feedback_matrix_CPU(self, T, G):
        """Build Wordle feedback matrix (CPU)."""
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

    def _build_wordle_feedback_matrix_GPU(self, T, G):
        """Build Wordle feedback matrix (GPU)."""
        # Encode words to GPU arrays
        key_words = cp.stack([cp.array([ord(c) - 97 for c in w], dtype=cp.int8) for w in T])
        all_words = cp.stack([cp.array([ord(c) - 97 for c in w], dtype=cp.int8) for w in G])

        K = key_words.shape[0]
        nG = all_words.shape[0]
        L = key_words.shape[1]  # 5 for Wordle

        # Compute letter counts for each target
        flat_idx = cp.ravel(key_words) + cp.repeat(cp.arange(K), L) * 26
        count_t = cp.bincount(flat_idx, minlength=K * 26).reshape(K, 26).astype(cp.int32)

        # Compute equality mask for greens
        equal = key_words[:, None, :] == all_words[None, :, :]

        # Initialize feedback
        feedback = cp.zeros((K, nG, L), dtype=cp.uint8)
        feedback[equal] = 2

        # Compute green counts per letter per pair
        green_counts = cp.zeros((K, nG, 26), dtype=cp.int32)
        for l in range(L):
            mask_l = equal[:, :, l]
            k, g = cp.where(mask_l)
            if len(k):
                c = key_words[k, l]
                green_counts[k, g, c] += 1

        # Remaining counts after greens
        remaining_counts = count_t[:, None, :] - green_counts

        # Second pass for yellows
        for i in range(L):
            mask = (feedback[:, :, i] == 0)
            letter_i = all_words[:, i]
            idx_k = cp.arange(K)[:, None]
            idx_g = cp.arange(nG)[None, :]
            idx_c = letter_i[None, :]
            rem = remaining_counts[idx_k, idx_g, idx_c]
            cond = (rem > 0) & mask
            feedback[:, :, i][cond] = 1
            remaining_counts[idx_k, idx_g, idx_c] -= cond.astype(cp.int32)

        # Compute the encoded feedback codes
        powers = cp.power(3, cp.arange(L - 1, -1, -1), dtype=cp.uint8)
        F = cp.sum(feedback * powers[None, None, :], axis=2, dtype=cp.uint8)
        return F

    @staticmethod
    def _wordle_decode_feedback(code):
        """Decode Wordle feedback code to tuple."""
        base, L = 3, 5
        out = []
        c = int(code)
        for i in range(L):
            out.append((c // (base ** (L - 1 - i))) % base)
        return tuple(out)

    def _read_csv(self, path):
        """Read CSV file for attribute-based instances."""
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

    def _build_compatibility_matrix(self):
        """
        Build compatibility matrix for constrained guessing mode.

        Determines which feedback codes are compatible: code i is compatible with j
        if every position in i is >= the corresponding position in j (elementwise).

        Currently hardcoded for 5-letter words (Wordle). For truly generic support,
        this would need to be configurable per instance.
        """
        xp = cp if self.use_gpu else np
        n = self.base
        L = 5  # Word length (hardcoded for Wordle)

        codes = xp.arange(n, dtype=xp.int32)
        digits = ((codes[:, None] // (3 ** xp.arange(L-1, -1, -1))) % 3).astype(xp.int8)

        # Compare all pairs (i, j): we want i >= j elementwise
        C = xp.all(digits[:, None, :] >= digits[None, :, :], axis=2)

        return C

    def get_instance_tuple(self):
        """
        Returns instance data as tuple for backward compatibility with existing code.

        Returns:
            (G_names, T_names, F, C, decode_feedback)
        """
        return (self.G_names, self.T_names, self.F, self.C, self.decode_feedback)

    def get_full_instance(self, flags, configs):
        """
        Build and return complete instance tuple with best guess functions.

        Args:
            flags: Runtime flags dict (print_diagnosis, evaluate, save_tree)
            configs: Runtime configs dict (metric, k, score)

        Returns:
            Complete instance tuple:
            (G_names, T_names, F, C, decode_feedback, best_guess_functions, best_guesses_functions)
        """
        # Merge instance properties into configs
        configs['base'] = self.base
        configs['is_target'] = self.is_target
        configs['guesses_include_targets'] = self.guesses_include_targets
        configs['constrained_guessing'] = self.constrained_guessing
        configs['GPU'] = self.use_gpu

        # Build instance tuple
        instance = self.get_instance_tuple()

        # Build best guess functions
        _best_guess_fns = best_guess_functions(instance, flags, configs)
        _best_guesses_fns = best_guesses_functions(configs)

        return instance + (_best_guess_fns, _best_guesses_fns)
