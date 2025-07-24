from .wordle_tools_utils import *


def get_instance():
    """
    Return instance for wordle solver model
    """
    words = _get_words("dataset/solutions.txt")
    num_of_letters = len(words[0])
    num_of_attempts = 6
    words_map = create_words_map(words)

    return words, num_of_letters, num_of_attempts, words_map


def fiter_instance(instance, guess_results):
    """
    Return filtered instances given guess results
    """
    words, num_of_letters, num_of_attempts, words_map = instance
    words_map = filter_words_map(words_map, guess_results)

    return words, num_of_letters, num_of_attempts, words_map


def _get_words(filepath):
    words = []

    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    return words

 
def get_guess_results(selected_word, word_guess):
    """
    Update game results for a single guess
    """
    if not word_guess:
        return []

    guess_results = []

    for i in range(len(selected_word)):
        if word_guess[i] == selected_word[i]:
            guess_results.append({'letter': word_guess[i], 'pos': i, 'status': 2})
        elif word_guess[i] in selected_word:
            guess_results.append({'letter': word_guess[i], 'pos': i, 'status': 1})
        else:
            guess_results.append({'letter': word_guess[i], 'pos': i, 'status': 0})

    return guess_results