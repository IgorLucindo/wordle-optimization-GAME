"""
Unified instance loader for the decision tree solver.

Encapsulates all instance loading, format handling, GPU conversion,
and setup logic in a single class.
"""
from pathlib import Path
from utils.xp_utils import cp, HAS_CUPY
from utils.guess_selection_utils import best_guess_functions, best_guesses_functions
from classes.feedback_engine import FeedbackEngine
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

        # Initialize feedback engine
        self.feedback_engine = FeedbackEngine(use_gpu=self.use_gpu)

        # Load raw data from disk
        data = self._load_from_disk(dataset_dir)

        # Extract and store instance properties
        self.G_names = data['G_names']
        self.T_names = data['T_names']
        self.decode_feedback = data['decode_feedback']

        # Base is now returned by the feedback engine
        self.base = data['base']
        rules = data['rules']
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

        # For constrained guessing mode, use full guess set for both T and G
        if rules.get('constrained_guessing', False):
            F, base = self.feedback_engine.build_matrix(G, G, rules)
        else:
            F, base = self.feedback_engine.build_matrix(T, G, rules)

        # First n_T guesses are targets
        is_target = np.zeros(n_G, dtype=bool)
        is_target[:n_T] = True

        return {
            'G_names': G,
            'T_names': T,
            'F': F,
            'base': base,
            'is_target': is_target,
            'decode_feedback': self.feedback_engine.get_decode_function(rules['feedback_engine']),
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

        feats_arr = np.array(uniq_feats, dtype=np.int16)
        n_attrs = feats_arr.shape[1]

        # Build feedback matrix using attribute engine
        F, base = self.feedback_engine.build_matrix(list(uniq_animals), list(attr_names), rules, raw_data=feats_arr)

        # Attributes are NOT targets (no self-identification)
        is_target = np.zeros(n_attrs, dtype=bool)

        return {
            'G_names': list(attr_names),
            'T_names': list(uniq_animals),
            'F': F,
            'base': base,
            'is_target': is_target,
            'decode_feedback': self.feedback_engine.get_decode_function(rules['feedback_engine']),
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

        # Build feedback matrix using mastermind engine
        F, base = self.feedback_engine.build_matrix(names, names, rules)

        # All codes can be targets (self-identifying)
        is_target = np.ones(n, dtype=bool)

        # Mastermind decode function needs pegs parameter
        def decode_feedback(code):
            code = int(code)
            pegs_plus_1 = pegs + 1
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

    @staticmethod
    def load_tree(filepath):
        """
        Load a saved decision tree from JSON file.

        Converts serialized format back to internal format:
        vertices: [(v_id, guess, is_terminal, depth), ...]
        successors: {(v_parent, feedback): v_child, ...}

        Args:
            filepath: Path to decision_tree.json file

        Returns:
            Tree dict with 'root', 'vertices', 'successors'
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert vertices from JSON format
        vertices = [(v, g, term, d) for v, g, term, d in data['vertices']]

        # Convert successors from JSON format (keys are strings "v_f")
        successors = {}
        for key_str, child_v in data['successors'].items():
            # Parse "v_f" format
            v_parent, feedback = map(int, key_str.split('_'))
            successors[(v_parent, feedback)] = child_v

        return {
            'root': data['root'],
            'vertices': vertices,
            'successors': successors
        }

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
