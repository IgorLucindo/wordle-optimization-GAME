from collections import deque
import numpy as np
import cupy as cp
import threading
import json
import time
import ast
import sys
import os


class Guess_Tree:
    def __init__(self, instance, flags, configs):
        # Get instance
        G, T, F, C, get_best_guess, get_best_guesses = instance
        xp = cp if configs['GPU'] else np
        self.xp = xp

        # Words as an int not strings
        self.words_map = G
        self.G = xp.arange(len(G))                    # Guess words
        self.T = xp.arange(len(T))                    # Target words
        self.F = F                                    # Feedback matrix
        self.C = C                                    # Feedback compatibility matrix
        self.get_best_guess = get_best_guess          # Best guess function
        self.get_best_guesses = get_best_guesses      # Best guesses function
        self.flags = flags
        self.configs = configs
        
        self.start_data = None
        self._stop_diagnosis = False
        self._diagnosis_thread = None
        self.tree = {
            'root': 1,
            'nodes': {},
            'successors': {}
        }
        self.decoded_tree = {}
        self.results = {
            'exp_guesses': 0,
            'std_guesses': 0,
            'max_guesses': 0,
            'distribution': None,
            'build_runtime': 0,
            'nodes': 0
        }
        self.node_count = 0
        self.depths = np.zeros(len(self.T))

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self, start_data=None, build_flag=True):
        """
        Iterative build using explicit queue (BFS)
        """
        xp = self.xp
        start_time = time.time()
        self.start_data = start_data
        G_hard = self.G if self.configs['hard_mode'] else None
        queue = deque([(self.T, G_hard, None, None, 1)])
        self.reset_tree()
        self.node_count = 0
        depths = []

        while queue:
            T_filtered, G_hard, parent_id, feedback, depth = queue.popleft()
            self.node_count += 1

            G_arg = G_hard if self.configs['hard_mode'] else self.G
            g_star, g_star_in_T = self.get_best_guess_starting_word(T_filtered, G_arg, self.F)
            self.append2Tree(g_star, self.node_count, parent_id, feedback, build_flag)
            if g_star_in_T:
                depths.append(depth)

            # Stop condition
            if len(T_filtered) == 1:
                continue
            
            # Partition candidates by feedback
            T_filtered = T_filtered[T_filtered != g_star]
            feedbacks = self.F[T_filtered, g_star]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, f_new in enumerate(unique_feedbacks):
                T_filtered_new = T_filtered[inverse_indices == i]
                G_hard_new = self.get_next_guesses_hardmode(T_filtered_new, G_hard, f_new, g_star)
                queue.append((T_filtered_new, G_hard_new, self.node_count, f_new.item(), depth + 1))

        self.depths = np.array(depths)
        self.results['build_runtime'] = time.time() - start_time
        self.results['nodes'] = self.node_count


    def get_best_guess_starting_word(self, T, G, F):
        """
        Get best guess considering a fixed starting word
        """
        if self.start_data is not None:
            g_star = self.start_data[0]
            g_star_in_T = self.start_data[1]
            self.start_data = None
        else:
            g_star, g_star_in_T = self.get_best_guess(T, G, F)
        
        return g_star, g_star_in_T


    def append2Tree(self, g_star, node_id, parent_id, feedback, build_flag):
        """
        Append node and edge to tree
        """
        if not build_flag:
            return
        
        self.tree['nodes'][node_id] = {'guess': g_star}
        self.tree['successors'][(parent_id, feedback)] = node_id


    def get_next_guesses_hardmode(self, T, G_hard, feedback, g_star):
        """
        Vectorized hard-mode filtering using precomputed LUT and feedback matrix
        Returns subset of allowed guess indices
        """
        if not self.configs['hard_mode'] or len(T) <= 2:
            return None
        
        # Feedbacks that each candidate (col) would produce w.r.t. previous guess (row)
        possible_feedbacks = self.F[G_hard, g_star]

        # Mask of which feedbacks are compatible
        valid_mask = self.C[possible_feedbacks, feedback]

        return G_hard[valid_mask]
    

    def reset_tree(self):
        """
        Resets the tree structure to its initial empty state
        """
        self.tree = {
            'root': 1,
            'nodes': {},
            'successors': {}
        }


    def decode_tree(self):
        """
        Decode tree
        """
        if not self.decoded_tree:
            return


    def load_tree(self, filepath='dataset/guess_tree.json'):
        """
        Loads the decision tree from a JSON file
        """
        with open(filepath, 'r') as f:
            tree = json.load(f)
            tree['nodes'] = {ast.literal_eval(k): v for k, v in tree['nodes'].items()}
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
            node_id = 1
            depth = 1

            while True:
                guess = self.tree['nodes'][node_id]['guess']

                if guess == target:
                    depths.append(depth)
                    break

                f = self.F[target, guess].item()
                node_id = self.tree['successors'][(node_id, f)]
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
    

    def print_results(self):
        """
        Prints the evaluation results to the console.
        """
        if not self.flags['evaluate']:
            return
        
        print(
            f"\n\n"
            f"Exp. guesses: {self.results['exp_guesses']:.3f}\n"
            f"Std. guesses: {self.results['std_guesses']:.3f}\n"
            f"Max. guesses: {self.results['max_guesses']}\n"
            f"Distribution: {self.results['distribution']}\n"
            f"Build Runtime: {self.results['build_runtime']:.3f}s\n"
            f"Nodes: {self.results['nodes']}\n"
            f"Best first word: {self.words_map[self.tree['nodes'][1]['guess'].item()]}\n"
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
                    f"\rNode count: {self.node_count} | Time: {elapsed}s   "
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

        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.decoded_tree, f)