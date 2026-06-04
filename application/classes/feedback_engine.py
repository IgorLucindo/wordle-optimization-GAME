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
                F = self._build_wordle_feedback_matrix(T, G)
            except Exception as e:
                # Check if it's a CUDA OOM error
                if 'out of memory' in str(e).lower() or 'OutOfMemoryError' in str(type(e).__name__):
                    print(f"⚠ GPU out of memory ({len(T)}×{len(G)} matrix). Falling back to CPU...")
                    old_gpu = self.use_gpu
                    self.use_gpu = False
                    F_np = self._build_wordle_feedback_matrix(T, G)
                    self.use_gpu = old_gpu
                    print("✓ CPU build complete. Converting to GPU...")
                    F = cp.array(F_np)
                else:
                    raise
        else:
            F = self._build_wordle_feedback_matrix(T, G)

        # Calculate base from word length: 3^L (ternary feedback per position)
        word_length = len(T[0])
        base = 3 ** word_length
        return F, base

    def _mastermind_engine(self, T, G, rules):
        """Mastermind black/white peg feedback engine."""
        xp = cp if self.use_gpu else np
        params = rules['generator_params']
        pegs = params['pegs']
        colors = params['colors']

        codes = [tuple(int(x) for x in name) for name in T]
        n = len(codes)
        codes_arr = xp.array(codes, dtype=xp.int8)

        # Black counts: F_black[i_t, i_g] = # positions where codes agree
        black = (codes_arr[:, None, :] == codes_arr[None, :, :]).sum(axis=2).astype(xp.int16)

        # Per-code color counts
        counts = xp.zeros((n, colors), dtype=xp.int16)
        for c in range(colors):
            counts[:, c] = (codes_arr == c).sum(axis=1)

        # Common multiset intersection
        common = xp.minimum(counts[:, None, :], counts[None, :, :]).sum(axis=2).astype(xp.int16)
        white = common - black

        pegs_plus_1 = pegs + 1
        F = (black * pegs_plus_1 + white).astype(xp.uint8)
        base = pegs_plus_1 * pegs_plus_1

        return F, base

    def _attribute_engine(self, T, G, raw_features):
        """Attribute matrix feedback engine."""
        xp = cp if self.use_gpu else np
        F = xp.array(raw_features, dtype=xp.uint8)
        max_val = int(raw_features.max())
        base = max(2, max_val + 1)
        return F, base

    def _build_wordle_feedback_matrix(self, T, G):
        """Build Wordle feedback matrix using xp (numpy or cupy based on use_gpu)."""
        xp = cp if self.use_gpu else np

        key_words = xp.stack([xp.array([ord(c) - 97 for c in w], dtype=xp.int8) for w in T])
        all_words = xp.stack([xp.array([ord(c) - 97 for c in w], dtype=xp.int8) for w in G])

        K = key_words.shape[0]
        nG = all_words.shape[0]
        L = key_words.shape[1]

        # Compute letter counts for each target
        flat_idx = (xp.ravel(key_words).astype(xp.int64) +
                    xp.repeat(xp.arange(K, dtype=xp.int64), L) * 26)
        count_t = xp.bincount(flat_idx, minlength=K * 26).reshape(K, 26).astype(xp.int32)

        # Compute equality mask for greens
        equal = key_words[:, None, :] == all_words[None, :, :]

        # Initialize feedback
        feedback = xp.zeros((K, nG, L), dtype=xp.uint8)
        feedback[equal] = 2

        # Compute green counts per letter per pair
        green_counts = xp.zeros((K, nG, 26), dtype=xp.int32)
        for l in range(L):
            mask_l = equal[:, :, l]
            k_idx, g_idx = xp.where(mask_l)
            if len(k_idx):
                c = key_words[k_idx, l]
                green_counts[k_idx, g_idx, c] += 1

        # Remaining counts after greens
        remaining_counts = count_t[:, None, :] - green_counts

        # Second pass for yellows
        for i in range(L):
            mask = (feedback[:, :, i] == 0)
            letter_i = all_words[:, i]
            idx_k = xp.arange(K, dtype=xp.int64)[:, None]
            idx_g = xp.arange(nG, dtype=xp.int64)[None, :]
            idx_c = letter_i[None, :].astype(xp.int64)
            rem = remaining_counts[idx_k, idx_g, idx_c]
            cond = (rem > 0) & mask
            feedback[:, :, i][cond] = 1
            remaining_counts[idx_k, idx_g, idx_c] -= cond.astype(xp.int32)

        # Encode feedback as powers of 3
        powers = xp.power(3, xp.arange(L - 1, -1, -1)).astype(xp.uint8)
        F = xp.sum(feedback * powers[None, None, :], axis=2, dtype=xp.uint8)
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
