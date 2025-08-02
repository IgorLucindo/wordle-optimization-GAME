from .instance_utils import *
from .calculation_utils import *
import random


def solve_random(instance, feedback):
    """
    Solve model for random word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance

    # Solve
    word_guess = random.choice(key_words)

    return word_guess


def solve_greedy(instance, feedback):
    """
    Solve model for optimal word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance
    
    # Solve
    word_guess = max(key_words, key=lambda word: get_word_probability(word, key_words))

    return word_guess


def solve_guess_tree(instance, feedback):
    """
    Solve model for trinary search word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance

    # Solve get tree node depending on feedback
    if not feedback:
        next_node = tree['root']
    else:
        current_node = tree['current_node']
        successors = tree['nodes'][current_node]['successors']
        next_node = successors[feedback]

    # Get word of node
    word_guess = tree['nodes'][next_node]['word']

    tree['current_node'] = next_node

    return word_guess