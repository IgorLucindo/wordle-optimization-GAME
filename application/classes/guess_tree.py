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
        self.node_count = 0
        self.build_runtime = -1

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self):
        """
        Iterative build using explicit queue (BFS)
        """
        xp = self.xp
        start_time = time.time()
        G_hard = self.G if self.configs['hard_mode'] else None
        queue = deque([(self.T, G_hard, None, None)])

        while queue:
            T_filtered, G_hard, parent_id, feedback = queue.popleft()
            self.node_count += 1

            G_arg = G_hard if self.configs['hard_mode'] else self.G
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


    def evaluate(self):
        """
        Evaluates the decision tree by simulating all possible games
        """
        if not self.flags['evaluate']:
            return

        depths = []
        hard_mode_valid = True
        
        for target in self.T:
            node_id = 1
            depth = 1
            prev_guesses = []
            prev_feedbacks = []

            while True:
                guess = self.tree['nodes'][node_id]['guess']

                if self.configs['hard_mode']:
                    if not self.is_valid_hardmode_guess(guess, prev_guesses, prev_feedbacks):
                        hard_mode_valid = False

                if guess == target:
                    depths.append(depth)
                    break

                f = self.F[target, guess].item()
                node_id = self.tree['successors'][(node_id, f)]
                depth += 1

        depths = np.array(depths)
        distribution = {int(d): int((depths == d).sum()) for d in np.unique(depths)}

        print(
            f"\n\nAverage guesses: {depths.mean():.3f}\n"
            f"Std guesses: {depths.std():.3f}\n"
            f"Max guesses: {depths.max()}\n"
            f"Distribution: {distribution}\n"
            f"Build Runtime: {self.build_runtime:.2f}s\n"
            f"Nodes: {self.node_count}\n"
        )
        if self.configs['hard_mode']:
            print(f"Hard mode valid: {hard_mode_valid}\n")


    def is_valid_hardmode_guess(self, guess_word, prev_guesses, prev_feedbacks):
        """
        Check if a guess satisfies Wordle hard mode rules based on previous feedbacks.
        Works on CPU (NumPy) since it’s lightweight.
        """
        # Convert encoded words to CPU numpy for simplicity
        xp = self.xp
        guess_word = xp.asnumpy(self.encoded_words[guess_word])
        
        for g_idx, fb in zip(prev_guesses, prev_feedbacks):
            prev_word = xp.asnumpy(self.encoded_words[g_idx])
            fb = int(fb)
            fb_pattern = np.base_repr(fb, base=3).zfill(len(guess_word))[-len(guess_word):]
            fb_pattern = np.array(list(fb_pattern), dtype=int)

            for pos, val in enumerate(fb_pattern):
                if val == 2:  # green
                    if guess_word[pos] != prev_word[pos]:
                        return False
                elif val == 1:  # yellow
                    # must contain the letter somewhere else, not same position
                    if prev_word[pos] not in guess_word or guess_word[pos] == prev_word[pos]:
                        return False
                elif val == 0:  # gray
                    # cannot contain this letter unless it's already confirmed yellow/green elsewhere
                    letter = prev_word[pos]
                    greens_yellows = prev_word[(fb_pattern == 1) | (fb_pattern == 2)]
                    if letter not in greens_yellows and letter in guess_word:
                        return False
        return True


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