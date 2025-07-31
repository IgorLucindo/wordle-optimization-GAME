from .wordle_tools_utils import *


def get_instance():
    """
    Return instance for wordle solver model
    """
    key_words = _get_words("dataset/solutions.txt")
    all_words = key_words + _get_words("dataset/non_solutions.txt")
    words = all_words.copy()
    num_of_letters = len(words[0])
    num_of_attempts = 6

    return all_words, words, key_words, num_of_letters, num_of_attempts


def fiter_instance(instance, guess_results):
    """
    Return filtered instances given guess results
    """
    all_words, words, key_words, num_of_letters, num_of_attempts = instance
    words = filter_words(words, guess_results)
    key_words = filter_words(key_words, guess_results)

    return all_words, words, key_words, num_of_letters, num_of_attempts


def _get_words(filepath):
    words = []

    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    return words

 
def get_guess_results(target_word, word_guess):
    """
    Update game results for a single guess
    """
    if not word_guess:
        return []

    guess_results = {'G': [], 'Y': [], 'B': []}

    for i in range(len(target_word)):
        if word_guess[i] == target_word[i]:
            guess_results['G'].append({'letter': word_guess[i], 'pos': i, 'status': 'G'})
        elif word_guess[i] in target_word:
            guess_results['Y'].append({'letter': word_guess[i], 'pos': i, 'status': 'Y'})
        else:
            guess_results['B'].append({'letter': word_guess[i], 'pos': i, 'status': 'B'})

    return guess_results