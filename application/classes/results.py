import json
import numpy as np


class Results:
    def __init__(self, instance, flags, configs):
        G, T, F, _, decode_feedback, _, _ = instance
        self.words_map = G
        self.T = np.arange(len(T))
        self.F = F
        self.decode_feedback = decode_feedback
        self.flags = flags
        self.configs = configs
        # Extract is_target array from configs
        self.is_target = configs.get('is_target')

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
        Evaluates the tree by simulating all possible games (Full Simulation).

        For games with self-id actions (Wordle, Mastermind): ``depth`` counts
        guesses and a leaf is reached when the vertex's action equals the
        target.

        For games without self-id actions (Zoo): the tree uses None at pure
        leaves, and the cost is the number of attribute queries on the
        root-to-leaf path (``depth - 1``).
        """
        D = []

        for target in self.T:
            v_curr = 0
            depth = 1

            while True:
                v_curr, guess = self.tree['vertices'][v_curr]
                # Normalize (guess may be a 0-d numpy / cupy array).
                if guess is None:
                    # Pure leaf (Zoo-style): identified by arrival
                    # Cost is number of queries = depth - 1
                    D.append(max(0, depth - 1))
                    break

                if hasattr(guess, 'item'):
                    guess_val = guess.item()
                else:
                    guess_val = int(guess)

                if guess_val == int(target):
                    # Target identified
                    D.append(depth)
                    break

                f = self.F[target, guess_val].item()
                v_curr = self.tree['successors'][(v_curr, f)]
                depth += 1

        self._calculate_stats(np.array(D))


    def evaluate_decoded(self):
        """
        Evaluates the decoded tree by simulating all possible games
        """
        # Check if any target can self-identify
        if self.is_target is None or not self.is_target.any():
            raise NotImplementedError("evaluate_decoded requires at least one self-identifying target")

        word_to_idx = {w: i for i, w in enumerate(self.words_map)}
        D = []

        for target in self.T:
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
                # Handle both array and tuple returns
                if isinstance(f_array, tuple):
                    f_tuple = f_array
                elif hasattr(f_array, 'tolist'):
                    f_tuple = tuple(f_array.tolist())
                else:
                    f_tuple = tuple(f_array)

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
            if guess_idx is None:
                # Pure leaf (Zoo): no terminal action
                self.decoded_tree['vertices'].append((v_curr, None))
            else:
                gi = guess_idx.item() if hasattr(guess_idx, 'item') else int(guess_idx)
                self.decoded_tree['vertices'].append(
                    (v_curr, self.words_map[gi])
                )

        # Decode Successors
        for (v_parent, feedback_code), child_id in self.tree['successors'].items():
            if feedback_code is None:
                continue

            if hasattr(feedback_code, 'item'):
                feedback_code = feedback_code.item()

            # Decode feedback
            feedback_array = self.decode_feedback(feedback_code)
            # Handle both array and tuple returns
            if isinstance(feedback_array, tuple):
                feedback_tuple = feedback_array
            elif hasattr(feedback_array, 'tolist'):
                feedback_tuple = tuple(feedback_array.tolist())
            else:
                feedback_tuple = tuple(feedback_array)

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
            root_action = self.tree['vertices'][0][1]
            if root_action is None:
                first_guess = "(leaf)"
            else:
                ra = root_action.item() if hasattr(root_action, 'item') else int(root_action)
                first_guess = self.words_map[ra]
        elif self.decoded_tree and self.decoded_tree['vertices']:
             first_guess = self.decoded_tree['vertices'][0][1] or "(leaf)"

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
        Saves the decoded tree to a JSON file in the instance-specific directory
        """
        if not self.flags['save_tree']:
            return

        self.decode_tree()

        # Save to instance-specific directory
        instance_name = self.configs.get('game', 'wordle')
        filepath = f"data/{instance_name}/decision_tree.json"

        with open(filepath, "w") as f:
            json.dump(self.decoded_tree, f)

        print(f"Tree saved in \"{filepath}\"\n")