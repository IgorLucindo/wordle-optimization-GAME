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
        G, T, F, get_best_guess = instance
        self.xp = cp if configs['GPU'] else np

        # Words as an int not strings
        self.words_map = G
        self.G = self.xp.arange(len(G))         # Guess words
        self.T = self.xp.arange(len(T))         # Target words
        self.F = F                              # Feedback matrix
        self.get_best_guess = get_best_guess    # Best guess function
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
        self.node_count = 0
        self.build_runtime = -1

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self):
        """
        Iterative build using explicit queue (BFS)
        """
        start_time = time.time()
        queue = deque([(self.T, None, None)])

        while queue:
            T_filtered, parent_id, feedback = queue.popleft() 

            self.node_count += 1
            node_id = self.node_count
            
            second_arg = T_filtered if self.configs['hard_mode'] else self.G
            word_guess = self.get_best_guess(T_filtered, second_arg, self.F)
            self.append2Tree(word_guess, node_id, parent_id, feedback)

            # Stop condition
            if len(T_filtered) == 1:
                continue
            
            # Partition candidates by feedback
            T_filtered = T_filtered[T_filtered != word_guess]
            feedbacks = self.F[T_filtered, word_guess]
            unique_feedbacks, inverse_indices = self.xp.unique(feedbacks, return_inverse=True)

            # Expand children
            for i, f_new in enumerate(unique_feedbacks):
                T_new_filtered = T_filtered[inverse_indices == i]
                queue.append((T_new_filtered, node_id, f_new.item()))

        self.build_runtime = time.time() - start_time
    

    def append2Tree(self, word_guess, node_id, parent_id, feedback):
        """
        Append node and edge to tree
        """
        self.tree['nodes'][node_id] = {'guess': word_guess}
        self.tree['edges'].append([parent_id, node_id])
        self.tree['successors'][(parent_id, feedback)] = node_id


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