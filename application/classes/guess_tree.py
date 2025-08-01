from utils.instance_utils import *
from utils.wordle_tools_utils import *
import numpy as np
import json
import time
import sys
import os


class Guess_Tree:
    def __init__(self, instance):
        all_words, words, key_words, num_of_letters, num_of_attempts = instance

        self.key_words = key_words
        self.all_words = all_words
        self.tree = {'root': None, 'nodes': {}, 'edges': []}
        self.node_count = 0
        self.start_time = time.time()

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self, filtered_key_words, previous_word_guess=None, feedback=None):
        """
        Recursively obtain the best guess tree
        """
        self.node_count += 1

        word_guess = self.get_best_guess(filtered_key_words)

        # Append node and edge to tree
        self.tree['nodes'][self.node_count] = {'word': word_guess, 'feedback': feedback}
        if self.node_count == 1:
            self.tree['root'] = word_guess
        else:
            self.tree['edges'].append([previous_word_guess, word_guess])

        if len(filtered_key_words) == 1:
            return

        # Branch to other nodes
        all_feedbacks = get_all_feedbacks(filtered_key_words, word_guess)
        for feedback in all_feedbacks:
            new_filtered = filter_words(filtered_key_words, word_guess, feedback)
            self.build(new_filtered, word_guess, feedback)


    def get_best_guess(self, filtered_key_words):
        """
        Return guess with min std len
        """
        best_w = None
        best_balance_score = float('inf')  # smaller is better

        for i, w in enumerate(self.all_words):
            # Print progress
            self.print_diagnosis(i, len(self.all_words))

            all_feedbacks = get_all_feedbacks(filtered_key_words, w)

            lengths = []
            for feedback in all_feedbacks:
                filtered = filter_words(filtered_key_words, w, feedback)
                lengths.append(len(filtered))
            
            # Measure imbalance — use standard deviation or range
            balance_score = np.mean(lengths)
            # balance_score = np.std(lengths)
            
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_w = w

        return best_w
    

    def print_diagnosis(self, word_num, total_words):
        """
        Print current word guess number and node count
        """
        if word_num % 10:
            return
        
        current_time = time.time() - self.start_time
        
        # Print
        sys.stdout.write(
            f"\rNode count: {self.node_count} | "
            f"Node Progress: {word_num * 100 / total_words:>.1f}% | "
            f"Time: {current_time:.2f}s         "
        )
        sys.stdout.flush()


    def save(self):
        with open(self.path + "guess_tree.json", "w") as f:
            json.dump(self.tree, f)
