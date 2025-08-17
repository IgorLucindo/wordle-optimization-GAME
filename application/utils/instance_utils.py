from utils.wordle_tools_utils import *
from utils.hash_utils import *
from collections import Counter
import json
import ast


def get_instance():
    """
    Return instance for wordle solver model
    """
    key_words = _get_words("dataset/solutions.txt")
    all_words = key_words + _get_words("dataset/non_solutions.txt")
    words = all_words.copy()
    num_of_letters = len(words[0])
    num_of_attempts = 6
    tree = _get_decision_tree()

    return all_words, words, key_words, num_of_letters, num_of_attempts, tree


def fiter_instance(instance, word_guess, feedback):
    """
    Return filtered instances given guess results
    """
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance
    words = filter_words(words, word_guess, feedback)
    key_words = filter_words(key_words, word_guess, feedback)

    return all_words, words, key_words, num_of_letters, num_of_attempts, tree


def _get_decision_tree():
    with open('dataset/decision_tree.json', 'r') as f:
        tree = json.load(f)
        tree['nodes'] = {ast.literal_eval(k): v for k, v in tree['nodes'].items()}

    return tree


def _get_words(filepath):
    words = []

    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    return words


def get_feedback(target_word, word_guess):
    """
    Update game results for a single guess
    """
    if not word_guess:
        return ""

    feedback = [''] * len(target_word)
    target_chars = list(target_word)
    guess_chars = list(word_guess)

    # Count letters for later 'present' (Y) check
    letter_counts = Counter(target_word)

    # First pass: Mark correct (G)
    for i in range(len(target_word)):
        if guess_chars[i] == target_chars[i]:
            feedback[i] = 'G'
            letter_counts[guess_chars[i]] -= 1  # Consume this letter

    # Second pass: Mark present (Y) and incorrect (B)
    for i in range(len(target_word)):
        if feedback[i] == 'G':
            continue
        if letter_counts[guess_chars[i]] > 0:
            feedback[i] = 'Y'
            letter_counts[guess_chars[i]] -= 1  # Consume this letter
        else:
            feedback[i] = 'B'

    return ''.join(feedback)


get_all_feedbacks_cache = {}
def get_all_feedbacks(key_words, word_guess):
    """
    Return all possible status results
    """
    # Memoization
    cache_key = (word_guess, hash_word_set(key_words))
    if cache_key in get_all_feedbacks_cache:
        return get_all_feedbacks_cache[cache_key]

    # Get all possible status combinations
    all_feedbacks = {get_feedback(key_word, word_guess) for key_word in key_words}

    # Storing for memoization
    get_all_feedbacks_cache[cache_key] = all_feedbacks

    return all_feedbacks