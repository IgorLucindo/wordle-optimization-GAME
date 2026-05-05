"""
Feedback engine for computing game feedback matrices.

Handles different feedback computation strategies (wordle ternary, mastermind pegs,
attribute matching) with CPU/GPU support.
"""
import numpy as np
from utils.xp_utils import cp


class FeedbackEngine:
    """
    Computes feedback matrices for different game types.

    Supports:
    - wordle_ternary: 5-position ternary feedback (green/yellow/gray)
    - mastermind_pegs: black/white peg counting
    - attribute_matrix: direct feature comparison
    """

    def __init__(self, use_gpu=False):
        """
        Initialize feedback engine.

        Args:
            use_gpu: Whether to use GPU acceleration (requires CuPy, wordle only)
        """
        self.use_gpu = use_gpu

    def build_matrix(self, T, G, rules, raw_data=None):
        """
        Build feedback matrix using appropriate engine.

        Args:
            T: Target names/codes
            G: Guess names/codes
            rules: Rules dictionary containing 'feedback_engine' key
            raw_data: Optional raw data for engines that need it (e.g., attribute matrix)

        Returns:
            For wordle: F matrix
            For mastermind/zoo: (F matrix, base)

        Raises:
            ValueError: If feedback_engine is unknown
        """
        engine = rules.get('feedback_engine')

        if engine == 'wordle_ternary':
            return self._wordle_engine(T, G)
        elif engine == 'mastermind_pegs':
            return self._mastermind_engine(T, G, rules)
        elif engine == 'attribute_matrix':
            return self._attribute_engine(T, G, raw_data)
        else:
            raise ValueError(f"Unknown feedback_engine: {engine}")

    def get_decode_function(self, engine_name):
        """
        Get the decode function for a given engine.

        Args:
            engine_name: Name of the feedback engine

        Returns:
            Decode function that converts feedback code to tuple
        """
        if engine_name == 'wordle_ternary':
            return self._wordle_decode_feedback
        elif engine_name == 'mastermind_pegs':
            return None  # Will be created dynamically by loader
        elif engine_name == 'attribute_matrix':
            return lambda code: (int(code),)
        else:
            raise ValueError(f"Unknown feedback_engine: {engine_name}")

    def _wordle_engine(self, T, G):
        """
        Wordle ternary feedback engine with GPU fallback.
        """
        if self.use_gpu:
            try:
                F = self._build_wordle_feedback_matrix_GPU(T, G)
            except Exception as e:
                # Check if it's a CUDA OOM error
                if 'out of memory' in str(e).lower() or 'OutOfMemoryError' in str(type(e).__name__):
                    print(f"⚠ GPU out of memory ({len(T)}×{len(G)} matrix). Falling back to CPU...")
                    F_cpu = self._build_wordle_feedback_matrix_CPU(T, G)
                    print("✓ CPU build complete. Converting to GPU...")
                    F = cp.array(F_cpu)
                else:
                    raise
        else:
            F = self._build_wordle_feedback_matrix_CPU(T, G)

        # Calculate base from word length: 3^L (ternary feedback per position)
        word_length = len(T[0])
        base = 3 ** word_length
        return F, base

    def _mastermind_engine(self, T, G, rules):
        """Mastermind black/white peg feedback engine."""
        params = rules['generator_params']
        pegs = params['pegs']
        colors = params['colors']

        codes = [tuple(int(x) for x in name) for name in T]
        n = len(codes)
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

        return F, base

    def _attribute_engine(self, T, G, raw_features):
        """Attribute matrix feedback engine."""
        F = raw_features.astype(np.uint8)
        max_val = int(raw_features.max())
        base = max(2, max_val + 1)
        return F, base

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
