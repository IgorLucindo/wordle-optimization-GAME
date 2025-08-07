from utils.instance_utils import *
from utils.wordle_tools_utils import *
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
        self.previous_time = 0

        self.score_cache = {}

        self.path = "application/results/"
        os.makedirs(self.path, exist_ok=True)


    def create(self):
        self.build(self.key_words)


    def build(self, filtered_key_words, previous_node_id=None, previous_feedback=None):
        """
        Recursively obtain the best decision tree
        """
        self.node_count += 1
        node_id = self.node_count

        # Get best guess
        if len(filtered_key_words) == 1:
            word_guess = filtered_key_words[0]
        else:
            word_guess, filtered_dict = self.get_best_guess_n_step(filtered_key_words, 1)

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
        for feedback, new_filtered in filtered_dict.items():
            self.build(new_filtered, node_id, feedback)
    

    def get_best_guess_n_step(self, filtered_key_words, depth):
        best_score = float('inf')
        best_w = None
        best_w_filtered_dict = {}

        for i, w in enumerate(self.all_words):
            self.print_diagnosis(i, len(self.all_words))

            score, filtered_dict = self.score_guess(w, filtered_key_words, depth)

            if score < best_score:
                best_score = score
                best_w = w
                best_w_filtered_dict = filtered_dict

        return best_w, best_w_filtered_dict
    

    def score_guess(self, guess_word, filtered_key_words, depth):
        """
        Recursively score a guess using N-step lookahead.
        Also returns the filtered_dict (feedback → filtered_words).
        """
        # Memoization
        cache_key = (guess_word, tuple(sorted(filtered_key_words)), depth)
        if cache_key in self.score_cache:
            print(1)
            return self.score_cache[cache_key]

        # Get filtered keywords for each feedback of guess word
        all_feedbacks = get_all_feedbacks(filtered_key_words, guess_word)
        filtered_dict = {
            f: filter_words(filtered_key_words, guess_word, f)
            for f in all_feedbacks
        }

        scores = []
        # weights = []
        for filtered in filtered_dict.values():
            if depth > 1 and len(filtered) == 1:
                scores.append(0)
                # weights.append(len(filtered))
                continue
            
            if depth > 1:
                best_sub_score = min(
                    self.score_guess(next_guess, filtered, depth - 1)[0]
                    for next_guess in self.all_words
                )
            else:
                best_sub_score = len(filtered)

            scores.append(best_sub_score)
            # weights.append(len(filtered))

        # Simple average
        avg_score = np.mean(scores) if scores else float('inf')

        # Weighted average
        # if weights and sum(weights) > 0:
        #     avg_score = sum(score * w for score, w in zip(scores, weights)) / sum(weights)
        # else:
        #     avg_score = float('inf')

        # Storing for memoization
        self.score_cache[cache_key] = (avg_score, filtered_dict)

        return avg_score, filtered_dict
    

    def print_diagnosis(self, word_num, total_words):
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
            f"Node Progress: {word_num * 100 / total_words:>.1f}% | "
            f"Time: {current_second}s   "
        )
        sys.stdout.flush()


    def save(self):
        with open(self.path + "decision_tree.json", "w") as f:
            json.dump(self.tree, f)
