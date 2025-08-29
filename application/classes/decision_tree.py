from utils.instance_utils import *
import numpy as np
import cupy as cp
import threading
import json
import time
import sys
import os


class Decision_Tree:
    def __init__(self, instance, flags):
        all_words, _, key_words, _, _, _ = instance

        # Words as an int not strings
        self.words_map = all_words
        self.key_words = cp.arange(len(key_words))
        self.all_words = cp.arange(len(all_words))
        self.FB = get_feedback_matrix(key_words, all_words)
        self.flags = flags
        
        self._stop_diagnosis = False
        self._diagnosis_thread = None
        self.tree = {
            'root': None,
            'nodes': {},
            'edges': []
        }
        self.node_count = 0
        self.previous_time = -1

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def create(self):
        """
        Iterative build using explicit stack (DFS)
        """
        stack = [(self.key_words, None, None)]

        while stack:
            filtered, parent_id, feedback = stack.pop()

            self.node_count += 1
            node_id = self.node_count
            
            word_guess = self.get_best_guess_normalized(filtered)
            self.append2Tree(word_guess, node_id, parent_id, feedback)

            # Stop condition
            if len(filtered) == 1:
                continue

            # Partition candidates by feedback
            feedbacks = self.FB[filtered, word_guess]
            unique_feedbacks, inverse_indices = cp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, fb in enumerate(unique_feedbacks):
                new_filtered = filtered[inverse_indices == i]
                stack.append((new_filtered, node_id, int(fb.item())))
    

    def get_best_guess(self, filtered_key_words):
        """
        Finds the best guess by maximizing the number of partitions (1-step lookahead).
        Special case:
        - If a candidate in filtered_key_words achieves exactly len(filtered_key_words) - 1 partitions,
            return that candidate.
        """
        n = len(filtered_key_words)
        if n <= 2:
            return filtered_key_words[0]

        feedbacks_sub = self.FB[filtered_key_words, :]
        pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
        unique_pairs = cp.unique(pairs)
        cols = unique_pairs // 243
        scores = cp.bincount(cols, minlength=feedbacks_sub.shape[1])

        # --- special case ---
        candidate_scores = scores[filtered_key_words]
        mask = candidate_scores == n
        if mask.any():
            # Pick the first candidate achieving this score
            return int(filtered_key_words[cp.argmax(mask)].item())

        # --- default case ---
        best_w = int(cp.argmax(scores).item())
        return best_w
    

    def get_best_guess_normalized(self, filtered_key_words):
        """
        Finds the best guess using a Pareto score:
        score = (#feedbacks) / (1 + std(new_filtered_lengths))
        """
        n = len(filtered_key_words)
        if n <= 2:
            return filtered_key_words[0]

        feedbacks_sub = self.FB[filtered_key_words, :]
        pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
        pairs_flat = pairs.ravel()
        uniq_pairs, counts = cp.unique(pairs_flat, return_counts=True)
        guess_idx = uniq_pairs // 243
        num_feedbacks = cp.bincount(guess_idx, minlength=feedbacks_sub.shape[1])
        sum_counts = cp.bincount(guess_idx, weights=counts, minlength=feedbacks_sub.shape[1])
        sum_counts_sq = cp.bincount(guess_idx, weights=counts**2, minlength=feedbacks_sub.shape[1])
        mean_counts = sum_counts / cp.maximum(num_feedbacks, 1)
        mean_counts_sq = sum_counts_sq / cp.maximum(num_feedbacks, 1)
        std_partition = cp.sqrt(cp.maximum(mean_counts_sq - mean_counts**2, 0))

        # Score normalized
        scores = num_feedbacks - 2*std_partition
        
        # --- special case ---
        candidate_scores = scores[filtered_key_words]
        mask = candidate_scores == n
        if mask.any():
            # Pick the first candidate achieving this score
            return int(filtered_key_words[cp.argmax(mask)].item())

        # Pick best guess
        best_w = int(cp.argmax(scores).item())

        return best_w
    

    def append2Tree(self, word_guess, node_id, previous_node_id, previous_feedback):
        """
        Append node and edge to tree
        """
        self.tree['nodes'][node_id] = {'word': word_guess, 'successors': {}}
        if node_id == 1:
            self.tree['root'] = node_id
        else:
            self.tree['edges'].append([previous_node_id, node_id])
            self.tree['nodes'][previous_node_id]['successors'][previous_feedback] = node_id


    def evaluate(self):
        """
        Evaluates the decision tree by simulating all possible games
        """
        if not self.flags['evaluate']:
            return

        depths = []
        
        for target in self.key_words:
            node_id = self.tree['root']
            depth = 1
            while True:
                node = self.tree['nodes'][node_id]
                guess = node['word']

                if guess == target:
                    depths.append(depth)
                    break

                fb = self.FB[target, guess].item()
                if fb not in node['successors']:
                    raise RuntimeError(f"Feedback path missing for target {target} at depth {depth}")
                node_id = node['successors'][fb]
                depth += 1

        depths = np.array(depths)
        distribution = {int(d): int((depths == d).sum()) for d in np.unique(depths)}

        print(
            f"\nAverage guesses: {depths.mean():.2f}\n"
            f"Std guesses: {depths.std():.2f}\n"
            f"Max guesses: {depths.max()}\n"
            f"Distribution: {distribution}\n"
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
        Saves the tree to a JSON file
        """
        if not self.flags['save']:
            return

        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.tree, f)
