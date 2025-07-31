from utils.wordle_tools_utils import *
import numpy as np
import sys


class Guess_Tree:
    def __init__(self, instance):
        all_words, words, key_words, num_of_letters, num_of_attempts = instance

        self.key_words = key_words
        self.all_words = all_words
        self.tree = {'root': None, 'nodes': [], 'edges': []}
        self.node_count = 0


    def build(self, filtered_key_words, previous_word_guess=None):
        """
        Recursively obtain the best guess tree
        """
        self.node_count += 1

        word_guess = self.get_best_guess(filtered_key_words)

        # Append node and edge to tree
        if self.node_count == 1:
            self.tree['root'] = word_guess
        else:
            self.tree['nodes'].append(word_guess)
            self.tree['edges'].append((previous_word_guess, word_guess))

        # Branch to other nodes
        all_guess_results = self.get_all_guess_results(word_guess, filtered_key_words)
        for guess_results in all_guess_results:
            filtered_key_words = filter_words(filtered_key_words, guess_results)

            if len(filtered_key_words) == 1:
                continue
            else:
                self.build(filtered_key_words, word_guess)


    def get_best_guess(self, filtered_key_words):
        """
        Return guess with min std len
        """
        best_w = None
        best_balance_score = float('inf')  # smaller is better

        for i, w in enumerate(self.all_words):
            self.print_diagnosis(i, len(self.all_words))
            all_guess_results = self.get_all_guess_results(w, filtered_key_words)

            lengths = []
            for guess_results in all_guess_results:
                filtered = filter_words(filtered_key_words, guess_results)
                lengths.append(len(filtered))
            
            # Measure imbalance — use standard deviation or range
            balance_score = np.std(lengths)  # could also use: max(lengths) - min(lengths)
            
            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_w = w

        return best_w
    

    def get_all_guess_results(self, word_guess, filtered_key_words):
        """
        Return all possible status results
        """
        all_status_combinations = set()
        all_guess_results = []

        # Get all possible status combinations
        for target_word in filtered_key_words:
            status_combination = self.get_status_combination(target_word, word_guess)
            all_status_combinations.add(status_combination)

        # Get all guess results based on all status combinations
        for status_combination in all_status_combinations:
            guess_results = {'G': [], 'Y': [], 'B': []}
            for i, s in enumerate(status_combination):
                guess_results[s].append({'letter': word_guess[i], 'pos': i, 'status': s})
            all_guess_results.append(guess_results)
                    
        return all_guess_results
    

    def get_status_combination(self, target_word, word_guess):
        """
        Return status combination of a guess given a target word
        """
        status_combination = []

        for i in range(len(target_word)):
            if word_guess[i] == target_word[i]:
                status_combination.append('G')
            elif word_guess[i] in target_word:
                status_combination.append('Y')
            else:
                status_combination.append('B')

        return tuple(status_combination)
    

    def print_diagnosis(self, word_num, total_words):
        """
        Print current word guess number and node count
        """
        sys.stdout.write(f"\rNode count: {self.node_count};      Word {word_num}/{total_words}")
        sys.stdout.flush()