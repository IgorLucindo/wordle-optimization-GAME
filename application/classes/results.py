import json
import numpy as np
import cupy as cp


class Results:
    def __init__(self, instance, flags, configs):
        G, T, F, _, decode_feedback, _, _ = instance
        self.words_map = G
        self.T = T
        self.F = F
        self.decode_feedback = decode_feedback
        self.flags = flags
        self.configs = configs

        # Result Containers
        self.tree = {'root': 0, 'vertices': [], 'successors': {}}
        self.decoded_tree = {}
        self.stats = {
            'exp_guesses': 0, 'std_guesses': 0, 'max_guesses': 0,
            'distribution': None, 'build_runtime': 0, '#vertices': 0
        }


    def set_data(self, tree, runtime):
        """
        Ingests the raw data from the solver
        """
        self.tree = tree
        self.stats['build_runtime'] = runtime
        self.stats['#vertices'] = len(tree['vertices'])


    def evaluate(self):
        """
        Evaluates the tree by simulating all possible games (Full Simulation)
        """
        targets = self.T if isinstance(self.T, (np.ndarray, list)) else cp.asnumpy(self.T)
        D = []

        for target in targets:
            v_curr = 0
            depth = 1

            while True:
                v_curr, guess = self.tree['vertices'][v_curr]
                if guess == target:
                    D.append(depth)
                    break
                f = self.F[target, guess].item()
                v_curr = self.tree['successors'][(v_curr, f)]
                depth += 1

        self._calculate_stats(np.array(D))


    def evaluate_decoded(self):
        """
        Evaluates the decoded tree by simulating all possible games
        """
        word_to_idx = {w: i for i, w in enumerate(self.words_map)}
        targets = self.T if isinstance(self.T, (np.ndarray, list)) else cp.asnumpy(self.T)
        D = []

        for target in targets:
            v_curr = 0
            depth = 1

            while True:
                v_curr, guess_str = self.decoded_tree['vertices'][v_curr]
                guess_idx = word_to_idx[guess_str]
                if guess_idx == target:
                    D.append(depth)
                    break
                f_code = self.F[target, guess_idx].item()
                f_array = self.decode_feedback(f_code)
                f_tuple = tuple(f_array.tolist())
                key = str((v_curr, f_tuple))
                v_curr = self.decoded_tree['successors'][key]
                if hasattr(v_curr, 'item'):
                    v_curr = v_curr.item()
                depth += 1

        self._calculate_stats(np.array(D))
        self.stats['#vertices'] = len(self.decoded_tree['vertices'])


    def _calculate_stats(self, D):
        distribution = {int(d): int((D == d).sum()) for d in np.unique(D)}
        self.stats['exp_guesses'] = D.mean()
        self.stats['std_guesses'] = D.std()
        self.stats['max_guesses'] = D.max()
        self.stats['distribution'] = distribution


    def decode_tree(self):
        """
        Decode tree: Converts internal IDs and codes to readable strings and tuples
        """
        if self.decoded_tree or not self.tree['vertices']:
            return
        
        self.decoded_tree = {'root': 0, 'vertices': [], 'successors': {}}

        # Decode vertices
        for v_curr, guess_idx in self.tree['vertices']:
            self.decoded_tree['vertices'].append(
                (v_curr, self.words_map[guess_idx.item()])
            )

        # Decode Successors
        for (v_parent, feedback_code), child_id in self.tree['successors'].items():
            if feedback_code is None:
                continue

            if hasattr(feedback_code, 'item'):
                feedback_code = feedback_code.item()
            
            # Decode feedback
            feedback_array = self.decode_feedback(feedback_code)
            feedback_tuple = tuple(feedback_array.tolist())
            key = str((v_parent, feedback_tuple))
            
            if hasattr(child_id, 'item'):
                child_id = child_id.item()

            self.decoded_tree['successors'][key] = child_id


    def load_tree(self, filepath):
        """
        Loads the decision tree from a JSON file
        """
        with open(filepath, 'r') as f:
            tree = json.load(f)
            self.decoded_tree = tree
            # If we load a tree, we often want to evaluate it immediately
            self.stats['#vertices'] = len(self.decoded_tree['vertices'])


    def print(self):
        """
        Prints the evaluation results to the console
        """
        if not self.flags['evaluate']:
            return
        
        # Determine first guess for display
        first_guess = "N/A"
        if self.tree['vertices']:
             first_guess = self.words_map[self.tree['vertices'][0][1].item()]
        elif self.decoded_tree and self.decoded_tree['vertices']:
             first_guess = self.decoded_tree['vertices'][0][1]

        print(
            f"\n\n"
            f"Exp. guesses: {self.stats['exp_guesses']:.3f}\n"
            f"Std. guesses: {self.stats['std_guesses']:.3f}\n"
            f"Max. guesses: {self.stats['max_guesses']}\n"
            f"Distribution: {self.stats['distribution']}\n"
            f"Build Runtime: {self.stats['build_runtime']:.3f}s\n"
            f"#Vertices: {self.stats['#vertices']}\n"
            f"Best first guess: {first_guess}\n"
        )


    def save(self):
        """
        Saves the decoded tree to a JSON file
        """
        if not self.flags['save_tree']:
            return
        
        self.decode_tree()
        filepath = (
            "results/decision_tree_hard.json"
            if self.configs['hard_mode']
            else "results/decision_tree.json"
        )

        with open(filepath, "w") as f:
            json.dump(self.decoded_tree, f)

        print(f"Tree saved in \"{filepath}\"\n")