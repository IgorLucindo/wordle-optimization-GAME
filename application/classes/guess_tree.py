from utils.instance_utils import *
from utils.wordle_tools_utils import *
import numpy as np
import json
import time
import sys
import os


class Guess_Tree:
    def __init__(self, instance, flags):
        all_words, words, key_words, num_of_letters, num_of_attempts = instance

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

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def build(self, filtered_key_words, previous_node_id=None, feedback=None):
        """
        Recursively obtain the best guess tree
        """
        self.node_count += 1
        current_node_id = self.node_count

        word_guess, all_feedbacks = self.get_best_guess(filtered_key_words)

        # Append node and edge to tree
        self.tree['nodes'][current_node_id] = {'word': word_guess, 'successors': {}}
        if current_node_id == 1:
            self.tree['root'] = current_node_id
        else:
            self.tree['edges'].append([previous_node_id, current_node_id])
            self.tree['nodes'][previous_node_id]['successors'][feedback] = current_node_id

        # Stop condition
        if len(filtered_key_words) == 1:
            return

        # Branch to other nodes
        for feedback in all_feedbacks:
            new_filtered = filter_words(filtered_key_words, word_guess, feedback)
            self.build(new_filtered, current_node_id, feedback)


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
                best_w_feedbacks = all_feedbacks

        return best_w, best_w_feedbacks
    

    def print_diagnosis(self, word_num, total_words):
        """
        Print current word guess number and node count
        """
        if word_num % 20 or not self.flags['print_diagnosis']:
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
