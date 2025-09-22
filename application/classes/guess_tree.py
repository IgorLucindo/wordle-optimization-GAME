from utils.instance_utils import *
import numpy as np
import cupy as cp
import threading
import json
import time
import sys
import os


class Guess_Tree:
    def __init__(self, instance, flags):
        # Get instance
        G, T, F, get_best_guess = instance

        # Words as an int not strings
        self.words_map = G
        self.T = cp.arange(len(T))              # Target words
        self.F = F                              # Feedback matrix
        self.get_best_guess = get_best_guess    # Best guess function
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
        self.build_runtime = -1

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def create(self):
        """
        Iterative build using explicit stack (DFS)
        """
        start_time = time.time()
        stack = [(self.T, None, None)]

        while stack:
            T_filtered, parent_id, feedback = stack.pop()

            self.node_count += 1
            node_id = self.node_count
            
            word_guess = self.get_best_guess(T_filtered, self.F)
            self.append2Tree(word_guess, node_id, parent_id, feedback)

            # Stop condition
            if len(T_filtered) == 1:
                continue
            
            # Partition candidates by feedback
            # T_filtered = T_filtered[T_filtered != word_guess]
            feedbacks = self.F[T_filtered, word_guess]
            unique_feedbacks, inverse_indices = cp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, f_new in enumerate(unique_feedbacks):
                T_new_filtered = T_filtered[inverse_indices == i]
                stack.append((T_new_filtered, node_id, int(f_new.item())))

        self.build_runtime = time.time() - start_time
    

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
        
        for target in self.T:
            node_id = self.tree['root']
            depth = 1
            while True:
                node = self.tree['nodes'][node_id]
                guess = node['word']

                if guess == target:
                    depths.append(depth)
                    break

                f = self.F[target, guess].item()
                if f not in node['successors']:
                    raise RuntimeError(f"Feedback path missing for target {target} at depth {depth}")
                node_id = node['successors'][f]
                depth += 1

        depths = np.array(depths)
        distribution = {int(d): int((depths == d).sum()) for d in np.unique(depths)}

        print(
            f"\n\nAverage guesses: {depths.mean():.2f}\n"
            f"Std guesses: {depths.std():.2f}\n"
            f"Max guesses: {depths.max()}\n"
            f"Distribution: {distribution}\n"
            f"Build Runtime: {self.build_runtime:.2f}s\n"
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
        if not self.flags['save_tree']:
            return

        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.tree, f)