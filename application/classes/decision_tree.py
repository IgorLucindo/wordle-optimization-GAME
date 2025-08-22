from utils.instance_utils import *
from utils.wordle_tools_utils import *
from utils.hash_utils import *
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
            
            word_guess = self.get_best_guess_2_step(filtered)
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
    

    def get_best_guess_1_step(self, filtered_key_words):
        """
        Finds the best guess by maximizing the number of partitions (1-step lookahead).
        """
        if len(filtered_key_words) <= 2:
            return filtered_key_words[0]
        
        feedbacks_sub = self.FB[filtered_key_words, :]
        pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
        unique_pairs = cp.unique(pairs)
        cols = unique_pairs // 243
        scores = cp.bincount(cols, minlength=feedbacks_sub.shape[1])
        best_w = int(cp.argmax(scores).item())

        return best_w
    

    def get_best_guess_2_step(self, filtered_key_words):
        """
        Finds the best guess by minimizing the worst-case (largest) partition size
        after the *next* guess (2-step lookahead).
        """
        if len(filtered_key_words) <= 2:
            return filtered_key_words[0]

        best_guess_A = -1
        # We want to minimize the max partition size, so we start with infinity.
        min_max_partition_size = float('inf')

        # 1. Outer loop: Iterate through every possible word as our first guess.
        for guess_A in self.all_words:
            # This will track the largest partition we might end up with after two guesses.
            worst_case_for_this_guess_A = 0

            # 2. Partition the candidates based on guess_A.
            feedbacks = self.FB[filtered_key_words, guess_A]
            unique_feedbacks, inverse_indices = cp.unique(feedbacks, return_inverse=True)

            # 3. Inner loop: For each resulting group, find the best next move.
            for i in range(len(unique_feedbacks)):
                partition = filtered_key_words[inverse_indices == i]

                # If the group is already solved, its future partition size is 1.
                if len(partition) <= 1:
                    max_sub_partition_size = len(partition)
                else:
                    # Find the best next guess (guess_B) for this specific group.
                    guess_B = self.get_best_guess_1_step(partition)
                    
                    # Find out how well guess_B splits this group.
                    sub_feedbacks = self.FB[partition, guess_B]
                    _, sub_counts = cp.unique(sub_feedbacks, return_counts=True)
                    
                    # The worst outcome for this branch is the largest sub-group.
                    max_sub_partition_size = int(cp.max(sub_counts).item())

                # Update the worst-case outcome for guess_A.
                if max_sub_partition_size > worst_case_for_this_guess_A:
                    worst_case_for_this_guess_A = max_sub_partition_size

            # 4. Minimax check: If this guess_A gives us a better "worst-case"
            # than the best one we've seen so far, it becomes our new best guess.
            if worst_case_for_this_guess_A < min_max_partition_size:
                min_max_partition_size = worst_case_for_this_guess_A
                best_guess_A = guess_A

        return best_guess_A
    

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
