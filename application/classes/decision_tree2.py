from utils.instance_utils import *
from utils.wordle_tools_utils import *
from utils.hash_utils import *
import numpy as np
import json
import time
import sys
import os


class Decision_Tree:
    def __init__(self, instance, flags):
        all_words, _, key_words, _, _, _ = instance

        self.key_words = key_words
        self.all_words = all_words
        self.flags = flags
        self.tree = {
            'root': None,
            'nodes': {},
            'edges': []
        }
        self.node_count = 0
        self.start_time = time.time()
        self.previous_time = -1

        self.score1_cache = {}
        self.score2_cache = {}

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def create(self):
        self.build(self.key_words)


    def build(self, filtered_key_words, previous_node_id=None, previous_feedback=None, previous_depth=0):
        """
        Recursively obtain the best decision tree
        """
        self.node_count += 1
        node_id = self.node_count
        depth = previous_depth + 1

        # Get best guess
        word_guess = self.get_best_guess(filtered_key_words, depth, _type=1)

        # Append node and edge to tree
        self.tree['nodes'][node_id] = {'word': word_guess, 'successors': {}}
        if node_id == 1:
            self.tree['root'] = node_id
        else:
            self.tree['edges'].append([previous_node_id, node_id])
            self.tree['nodes'][previous_node_id]['successors'][previous_feedback] = node_id

        # Stop condition
        if len(filtered_key_words) == 1:
            return

        # Branch to other nodes
        all_feedbacks = get_all_feedbacks(filtered_key_words, word_guess)
        for f in all_feedbacks:
            new_filtered = filter_words(filtered_key_words, word_guess, f)
            self.build(new_filtered, node_id, f, depth)
    

    def get_best_guess(self, filtered_key_words, depth=1, _type=2):
        if len(filtered_key_words) == 1 or len(filtered_key_words) == 2:
            return filtered_key_words[0]

        best_score = float('inf')
        best_w = None

        for i, w in enumerate(self.all_words):
            if _type == 1:
                self.print_diagnosis(i)
                score = self.score1_guess(w, filtered_key_words, depth)
            else:
                score = self.score2_guess(w, filtered_key_words)

            if score < best_score:
                best_score = score
                best_w = w

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
    

    def score2_guess(self, word_guess, filtered_key_words):
        """
        Return word with best score2
        """
        # Memoization
        cache_key = (word_guess, hash_word_set(filtered_key_words))
        if cache_key in self.score2_cache:
            return self.score2_cache[cache_key]

        scores = []

        all_feedbacks = get_all_feedbacks(filtered_key_words, word_guess)
        for f in all_feedbacks:
            filtered = filter_words(filtered_key_words, word_guess, f)
            scores.append(len(filtered))

        # Simple average
        avg_score = np.mean(scores) if scores else float('inf')

        # Storing for memoization
        self.score2_cache[cache_key] = avg_score

        return avg_score
    

    def print_diagnosis(self, word_num):
        """
        Print current word guess number and node count
        """
        current_time = time.time() - self.start_time
        current_second = int(current_time)

        if current_second == self.previous_time or not self.flags['print_diagnosis']:
            return
        
        self.previous_time = current_second
        
        # Print
        sys.stdout.write(
            f"\rNode count: {self.node_count} | "
            f"Node Progress: {word_num * 100 / len(self.all_words):>.1f}% | "
            f"Time: {current_second}s   "
        )
        sys.stdout.flush()


    def save(self):
        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.tree, f)
