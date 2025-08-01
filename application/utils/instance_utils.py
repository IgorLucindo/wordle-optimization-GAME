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


def fiter_instance(instance, word_guess, feedback):
    """
    Return filtered instances given guess results
    """
    all_words, words, key_words, num_of_letters, num_of_attempts = instance
    words = filter_words(words, word_guess, feedback)
    key_words = filter_words(key_words, word_guess, feedback)

    return all_words, words, key_words, num_of_letters, num_of_attempts


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

    feedback = []

    for i in range(len(target_word)):
        if word_guess[i] == target_word[i]:
            feedback.append('G')
        elif word_guess[i] in target_word:
            feedback.append('Y')
        else:
            feedback.append('B')

    return "".join(feedback)


def get_all_feedbacks(key_words, word_guess):
    """
    Return all possible status results
    """
    all_feedbacks = set()

    # Get all possible status combinations
    for target_word in key_words:
        feedback = get_feedback(target_word, word_guess)
        all_feedbacks.add(feedback)

    return list(all_feedbacks)