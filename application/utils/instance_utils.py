from utils.wordle_tools_utils import *


def get_instance(game_results):
    """
    Return instance for wordle solver model
    """
    all_words = _get_all_words()
    num_of_letters = len(all_words[0])
    num_of_attempts = 6
    words_map = create_words_map(all_words)
    words_map = filter_words_map(words_map, game_results)

    return all_words, num_of_letters, num_of_attempts, words_map


def _get_all_words():
    solution_words = _get_words("dataset/solutions.txt")
    non_solution_words = _get_words("dataset/non_solutions.txt")

    return solution_words + non_solution_words


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