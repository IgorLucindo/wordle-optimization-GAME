from utils.hash_utils import *
import random


def get_random_word(words):
    return random.choice(words)


filter_words_cache = {}
def filter_words(words, word_guess, feedback):
    """
    Handle guess results in order to update possible words
    """
    if not feedback:
        return words
    
    # Memoization
    cache_key = (word_guess, hash_word_set(words), feedback)
    if cache_key in filter_words_cache:
        return filter_words_cache[cache_key]

    # Count how many times the letter appears
    letter_count = {
        letter: sum(
            1 for pos, fb in enumerate(feedback)
            if word_guess[pos] == letter and fb != 'B'
        )
        for letter in word_guess
    }

    # Handle each letter result
    for pos, fb in enumerate(feedback):
        if fb == 'G':
            words = handle_correct_status(words, word_guess[pos], pos)
    for pos, fb in enumerate(feedback):
        if fb == 'B':
            words = handle_incorrect_status(words, word_guess[pos], pos, letter_count)
    for pos, fb in enumerate(feedback):
        if fb == 'Y':
            words = handle_present_status(words, word_guess[pos], pos)
  
    # Storing for memoization
    filter_words_cache[cache_key] = words

    return words


def handle_correct_status(words, letter, pos):
    """
    Letter is in the correct position
    """
    words_with_letter_pos = [
        w
        for w in words
        if w[pos] == letter
    ]

    return words_with_letter_pos


def handle_present_status(words, letter, pos):
    """
    Letter is in the word but NOT at this position
    """
    words_with_letter_elsewhere = [
        w
        for w in words
        if letter in w and w[pos] != letter
    ]

    return words_with_letter_elsewhere


def handle_incorrect_status(words, letter, pos, letter_count):
    """
    Letter appears in the word a defined amount of times
    """
    words_with_num_of_letters = [
        word
        for word in words
        if word.count(letter) <= letter_count[letter] and word[pos] != letter
    ]
    
    return words_with_num_of_letters