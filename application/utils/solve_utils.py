from .instance_utils import *
from .calculation_utils import *
import random


def solve_random(instance):
    """
    Solve model for random word guess
    """
    # Unpack data
    words, num_of_letters, num_of_attempts = instance

    # Solve
    word_guess = random.choice(words)

    return word_guess


def solve_optimal(instance):
    """
    Solve model for optimal word guess
    """
    # Unpack data
    words, num_of_letters, num_of_attempts = instance
    
    # Solve
    word_guess = max(words, key=lambda word: get_word_probability(word, words))

    return word_guess