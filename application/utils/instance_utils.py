from utils.wordle_tools_utils import *


def get_instance():
    """
    Return instance for wordle solver model
    """
    words = _get_words("dataset/solutions.txt")
    num_of_letters = len(words[0])
    num_of_attempts = 6
    words_map = create_words_map(words)

    return words, num_of_letters, num_of_attempts, words_map


def fiter_instance(instance, game_results):
    """
    Return filtered instances given game results
    """
    words, num_of_letters, num_of_attempts, words_map = instance
    words_map = filter_words_map(words_map, game_results)

    return words, num_of_letters, num_of_attempts, words_map


def _get_words(filepath):
    words = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                word = line.strip()
                words.append(word)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")

    return words