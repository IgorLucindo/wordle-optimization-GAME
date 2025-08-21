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
        self.score1_cache = {}

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def create(self):
        """
        Iterative build using explicit stack (DFS)
        """
        stack = [(self.key_words, None, None, 0)]

        while stack:
            filtered, parent_id, feedback, depth = stack.pop()

            self.node_count += 1
            node_id = self.node_count
            
            word_guess = self.get_best_guess2(filtered)
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
                stack.append((new_filtered, node_id, int(fb.item()), depth + 1))
    

    def get_best_guess1(self, filtered_key_words, depth=1):
        if len(filtered_key_words) <= 2:
            return filtered_key_words[0]

        best_score = float('-inf')
        best_w = None

        for w in self.all_words:
            score = self.score1_guess(w, filtered_key_words, depth)

            if score > best_score:
                best_score = score
                best_w = w
                
        return best_w
    

    def get_best_guess2(self, filtered_key_words):
        if len(filtered_key_words) <= 2:
            return filtered_key_words[0]
        
        feedbacks_sub = self.FB[filtered_key_words, :]
        pairs = feedbacks_sub + cp.arange(feedbacks_sub.shape[1]) * 243
        unique_pairs = cp.unique(pairs)
        cols = unique_pairs // 243
        scores = cp.bincount(cols, minlength=feedbacks_sub.shape[1])
        best_w = int(cp.argmax(scores).item())

        return best_w
    

    def score1_guess(self, word_guess, filtered_key_words, depth):
        """
        Compute average number of guesses if starting with `word_guess`,
        then building the rest of the tree using score2.
        """
        cache_key = (word_guess, hash_word_set(filtered_key_words))
        if cache_key in self.score1_cache:
            return self.score1_cache[cache_key]

        total_guesses = 0
        for key_word in filtered_key_words:
            # Simulate a full game with this target
            guesses = self.simulate_game(word_guess, key_word, filtered_key_words, depth)

            # Stop if didn't solve one keyword
            if guesses == None:
                return float('inf')

            total_guesses += guesses

        avg_guesses = total_guesses / len(filtered_key_words)
        self.score1_cache[cache_key] = avg_guesses
        return avg_guesses


    def simulate_game(self, word_guess, key_word, filtered_key_words, depth):
        """
        Simulate playing Wordle with a fixed target word.
        First guess is fixed, then we use score2 to choose subsequent guesses.
        """
        guesses = depth
        candidates = filtered_key_words

        while word_guess != key_word:
            feedback = get_feedback(key_word, word_guess)
            candidates = filter_words(candidates, word_guess, feedback)

            # Stop condition
            if guesses == 5 and len(candidates) > 1:
                return None
            
            word_guess = self.get_best_guess(candidates, _type=2)
            guesses += 1
 
        return guesses
    

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
