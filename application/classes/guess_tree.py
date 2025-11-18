from utils.instance_utils import *
from collections import deque
import numpy as np
import cupy as cp
import threading
import json
import time
import sys
import os


class Guess_Tree:
    def __init__(self, instance, flags, configs):
        # Get instance
        G, T, F, GGF, get_best_guess, feedback_compat = instance
        self.xp = cp if configs['GPU'] else np
        xp = self.xp

        # Words as an int not strings
        self.words_map = G
        self.encoded_words = xp.array([[ord(c) - ord('a') for c in w] for w in self.words_map], dtype=xp.int8)
        self.G = xp.arange(len(G))              # Guess words
        self.T = xp.arange(len(T))              # Target words
        self.F = F                              # Feedback matrix
        self.GGF = GGF                          # Feedback matrix
        self.get_best_guess = get_best_guess    # Best guess function
        self.feedback_compat = feedback_compat
        self.flags = flags
        self.configs = configs
        
        self._stop_diagnosis = False
        self._diagnosis_thread = None
        self.tree = {
            'root': 1,
            'nodes': {},
            'edges': [],
            'successors': {}
        }
        self.results = {}
        self.node_count = 0
        self.tree_count = 0
        self.build_runtime = -1

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self, starting_word=None):
        """
        Iterative build using explicit queue (BFS)
        """
        xp = self.xp
        start_time = time.time()
        G_hard = self.G if self.configs['hard_mode'] else None
        queue = deque([(self.T, G_hard, None, None)])
        self.reset_tree()
        self.node_count = 0

        while queue:
            T_filtered, G_hard, parent_id, feedback = queue.popleft()
            self.node_count += 1

            G_arg = G_hard if self.configs['hard_mode'] else self.G
            if self.node_count == 1 and starting_word:
                word_guess = starting_word
            else:
                word_guess = self.get_best_guess(T_filtered, G_arg, self.F)
            self.append2Tree(word_guess, self.node_count, parent_id, feedback)

            # Stop condition
            if len(T_filtered) == 1:
                continue
            
            # Partition candidates by feedback
            T_filtered = T_filtered[T_filtered != word_guess]
            feedbacks = self.F[T_filtered, word_guess]
            unique_feedbacks, inverse_indices = xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, f_new in enumerate(unique_feedbacks):
                T_filtered_new = T_filtered[inverse_indices == i]
                G_hard_new = self.get_next_guesses_hardmode(T_filtered_new, G_hard, f_new, word_guess)
                queue.append((T_filtered_new, G_hard_new, self.node_count, f_new.item()))

        self.build_runtime = time.time() - start_time
    

    def build_for_all_words(self):
        """
        Builds a decision tree for each possible starting word and evaluates it
        """
        results_pairs = []

        for g in self.G:
            self.build(g)
            self.tree_count += 1
            self.evaluate()
            
            # Store the average guesses along with the word 'g'
            avg = self.results['avg_guesses']
            first_guess = self.words_map[g.item()]
            results_pairs.append((avg, first_guess))

        # Final result processing
        results_pairs.sort()
        
        # Print results
        print("\n\nScores:")
        for pair in results_pairs:
            print(f"{pair[1]}: {pair[0]:.3f}")


    def append2Tree(self, word_guess, node_id, parent_id, feedback):
        """
        Append node and edge to tree
        """
        self.tree['nodes'][node_id] = {'guess': word_guess}
        self.tree['edges'].append([parent_id, node_id])
        self.tree['successors'][(parent_id, feedback)] = node_id


    def get_next_guesses_hardmode(self, T, G_hard, feedback, word_guess):
        """
        Vectorized hard-mode filtering using precomputed LUT and GGF matrix
        Returns subset of allowed guess indices
        """
        if not self.configs['hard_mode'] or len(T) <= 2:
            return None
        
        # Feedbacks that each candidate (col) would produce w.r.t. previous guess (row)
        possible_feedbacks = self.GGF[G_hard, word_guess]

        # Mask of which feedbacks are compatible
        valid_mask = self.feedback_compat[possible_feedbacks, feedback]

        return G_hard[valid_mask]
    

    def reset_tree(self):
        """
        Resets the tree structure to its initial empty state
        """
        self.tree = {
            'root': 1,
            'nodes': {},
            'edges': [],
            'successors': {}
        }


    def evaluate(self):
        """
        Evaluates the decision tree by simulating all possible games
        """
        if not self.flags['evaluate']:
            return

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

        self.results = {
            'avg_guesses': depths.mean(),
            'std_guesses': depths.std(),
            'max_guesses': depths.max(),
            'distribution': distribution,
            'build_runtime': self.build_runtime,
            'nodes': self.node_count
        }
    

    def print_results(self):
        """
        Prints the evaluation results to the console.
        """
        if not self.flags['evaluate']:
            return
        
        print(
            f"\n\nAverage guesses: {self.results['avg_guesses']:.3f}\n"
            f"Std guesses: {self.results['std_guesses']:.3f}\n"
            f"Max guesses: {self.results['max_guesses']}\n"
            f"Distribution: {self.results['distribution']}\n"
            f"Build Runtime: {self.results['build_runtime']:.2f}s\n"
            f"Nodes: {self.results['nodes']}\n"
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
                    f"\rTree count: {self.tree_count} | Node count: {self.node_count} | Time: {elapsed}s   "
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
        Saves the tree to a JSON file
        """
        if not self.flags['save_tree']:
            return

        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.tree, f)