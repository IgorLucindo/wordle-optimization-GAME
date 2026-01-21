from classes.device_optimizer import DeviceOptimizer
from collections import deque
import numpy as np
import threading
import json
import time
import sys


class Guess_Tree:
    def __init__(self, instance, flags, configs):
        # Initialize Optimizer (Handles calibration, data, and solvers)
        self.optimizer = DeviceOptimizer(instance, flags, configs)
        
        G, T, F, C, decode_feedback, _, _ = instance
        self.xp = self.optimizer.xp

        # Words as an int not strings
        self.words_map = G
        self.G = self.xp.arange(len(G)) # Guess words
        self.T = self.xp.arange(len(T)) # Target words
        self.F = F                      # Feedback matrix (Default Device)
        self.C = C                      # Feedback compatibility matrix
        self.decode_feedback = decode_feedback
        self.flags = flags
        self.configs = configs
        
        self.start_data = None
        self._stop_diagnosis = False
        self._diagnosis_thread = None
        self.tree = {'root': 0, 'vertices': [], 'successors': {}}
        self.decoded_tree = {}
        self.results = {
            'exp_guesses': 0,
            'std_guesses': 0,
            'max_guesses': 0,
            'distribution': None,
            'build_runtime': 0,
            '#vertices': 0
        }
        self.v_curr = -1
        self.depths = np.zeros(len(self.T))


    def build_tree(self):
        """
        Build tree iteratively using explicit queue (BFS)
        """
        start_time = time.time()

        # Queue: (T_curr, G_curr, v_parent, p_parent, depth)
        G_curr = self.G if self.configs['hard_mode'] else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        
        self.reset_tree()
        self.v_curr = -1
        depths = []

        while queue:
            T_curr, G_curr, v_parent, p_parent, depth = queue.popleft()
            self.v_curr += 1
            
            # Ask optimizer for the best context for this specific vertex
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            G_arg = G_curr if self.configs['hard_mode'] else self.G
            g_star, g_star_in_T = get_best_guess(T_curr, G_arg, F)
            
            self.append2Tree(g_star, self.v_curr, v_parent, p_parent)
            if g_star_in_T:
                depths.append(depth)

            # Stop condition
            if len(T_curr) == 1:
                continue
            
            # Partition candidates by feedback
            T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self.get_next_guesses_hardmode(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        self.depths = np.array(depths)
        self.results['build_runtime'] = time.time() - start_time
        self.results['#vertices'] = self.v_curr


    def build_subtree(self, g_start, g_start_in_T):
        """
        Build subtree iteratively using explicit queue (BFS)
        """
        start_time = time.time()

        # Queue: (T_curr, G_curr, v_parent, p_parent, depth)
        G_curr = self.G if self.configs['hard_mode'] else None
        queue = deque([(self.T, G_curr, -1, None, 1)])
        self.v_curr = -1
        depths = []

        while queue:
            T_curr, G_curr, v_parent, p_parent, depth = queue.popleft()
            self.v_curr += 1
            
            # Ask optimizer for the best context for this specific vertex
            T_curr, G_curr, xp, F, C, get_best_guess, _ = self.optimizer.get_context(T_curr, G_curr)

            # Get best guess considering starting guess
            if g_start is not None:
                g_star, g_star_in_T = g_start, g_start_in_T
                g_start = None
            else:
                G_arg = G_curr if self.configs['hard_mode'] else self.G
                g_star, g_star_in_T = get_best_guess(T_curr, G_arg, F)

            if g_star_in_T:
                depths.append(depth)

            # Stop condition
            if len(T_curr) == 1:
                continue
            
            # Partition candidates by feedback
            T_curr = T_curr[T_curr != g_star]
            feedbacks = F[T_curr, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, p in enumerate(unique_feedbacks):
                T_p = T_curr[inverse_indices == i]
                G_p = self.get_next_guesses_hardmode(T_p, G_curr, p, g_star, F, C)
                queue.append((T_p, G_p, self.v_curr, p.item(), depth + 1))

        self.depths = np.array(depths)
        self.results['build_runtime'] = time.time() - start_time
        self.results['#vertices'] = self.v_curr


    def append2Tree(self, g_star, v_curr, v_parent, p_parent):
        """
        Append vertex and edge to tree
        """
        self.tree['vertices'].append((v_curr, g_star))
        if v_curr != 0:
            self.tree['successors'][(v_parent, p_parent)] = v_curr


    def get_next_guesses_hardmode(self, T, G, feedback, g_star, F, C):
        """
        Vectorized hard-mode filtering using precomputed LUT and feedback matrix
        Returns subset of allowed guess indices
        """
        if not self.configs['hard_mode'] or len(T) <= 2:
            return None
        
        # Feedbacks that each candidate (col) would produce w.r.t. previous guess (row)
        possible_feedbacks = F[G, g_star]

        # Mask of which feedbacks are compatible
        valid_mask = C[possible_feedbacks, feedback]

        return G[valid_mask]
    

    def reset_tree(self):
        """
        Resets the tree structure to its initial empty state
        """
        self.tree = {'root': 0, 'vertices': [], 'successors': {}}


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


    def evaluate_quick(self):
        """
        Evaluates the tree by deriving data from its depths
        """
        self.results['exp_guesses'] = self.depths.mean()
        self.results['std_guesses'] = self.depths.std()
        self.results['max_guesses'] = self.depths.max()


    def evaluate(self):
        """
        Evaluates the tree by simulating all possible games
        """
        depths = []
        
        for target in self.T:
            v_curr = 0
            depth = 1

            while True:
                v_curr, guess = self.tree['vertices'][v_curr]
                if guess == target:
                    depths.append(depth)
                    break
                f = self.F[target, guess].item()
                v_curr = self.tree['successors'][(v_curr, f)]
                depth += 1

        depths = np.array(depths)
        distribution = {int(d): int((depths == d).sum()) for d in np.unique(depths)}

        self.results['exp_guesses'] = depths.mean()
        self.results['std_guesses'] = depths.std()
        self.results['max_guesses'] = depths.max()
        self.results['distribution'] = distribution


    def evaluate_decoded(self):
        """
        Evaluates the decoded tree by simulating all possible games
        """
        depths = []
        word_to_idx = {w: i for i, w in enumerate(self.words_map)}

        for target in self.T:
            v_curr = 0
            depth = 1

            while True:
                v_curr, guess_str = self.decoded_tree['vertices'][v_curr]
                guess_idx = word_to_idx[guess_str]
                if guess_idx == target:
                    depths.append(depth)
                    break
                f_code = self.F[target, guess_idx].item()
                f_array = self.decode_feedback(f_code)
                f_tuple = tuple(f_array.tolist())
                key = str((v_curr, f_tuple))
                v_curr = self.decoded_tree['successors'][key]
                if hasattr(v_curr, 'item'):
                    v_curr = v_curr.item()
                depth += 1

        depths = np.array(depths)
        distribution = {int(d): int((depths == d).sum()) for d in np.unique(depths)}

        self.results['exp_guesses'] = depths.mean()
        self.results['std_guesses'] = depths.std()
        self.results['max_guesses'] = depths.max()
        self.results['distribution'] = distribution
        self.results['#vertices'] = len(self.decoded_tree['vertices'])
    

    def print_results(self):
        """
        Prints the evaluation results to the console
        """
        if not self.flags['evaluate']:
            return
        
        first_guess = (
            self.words_map[self.tree['vertices'][0][1].item()] 
            if self.tree['vertices']
            else self.decoded_tree['vertices'][0][1]
        )
        
        print(
            f"\n\n"
            f"Exp. guesses: {self.results['exp_guesses']:.3f}\n"
            f"Std. guesses: {self.results['std_guesses']:.3f}\n"
            f"Max. guesses: {self.results['max_guesses']}\n"
            f"Distribution: {self.results['distribution']}\n"
            f"Build Runtime: {self.results['build_runtime']:.3f}s\n"
            f"#Vertices: {self.results['#vertices']}\n"
            f"Best first guess: {first_guess}\n"
        )


    def start_diagnosis(self):
        """
        Starts a diagnosis thread to print information about the tree building process
        """
        if not self.flags['print_diagnosis']:
            return
        
        start_time = time.time()

        def _diagnose():
            while not self._stop_diagnosis:
                elapsed = int(time.time() - start_time)
                sys.stdout.write(
                    f"\rVertex count: {self.v_curr + 1} | Time: {elapsed}s   "
                )
                sys.stdout.flush()
                time.sleep(1)

        self._diagnosis_thread = threading.Thread(target=_diagnose, daemon=True)
        self._diagnosis_thread.start()


    def stop_diagnosis(self):
        """
        Stops the diagnosis thread
        """
        self._stop_diagnosis = True
        if self._diagnosis_thread is not None:
            self._diagnosis_thread.join()


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