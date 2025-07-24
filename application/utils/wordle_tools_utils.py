from collections import defaultdict
import random


def get_random_word(words):
    return random.choice(words)


def create_words_map(words):
    """
    Return instance for wordle solver model
    """
    words_map = defaultdict(lambda: defaultdict(list))

    for word in words:
        for j in range(len(word)):
            words_map[word[j]][j].append(word)

    return words_map


def flatten_words_map(words_map):
    """
    Return list of flattened words map dictionary
    """
    flattened_words = list({
        element
        for inner_dict in words_map.values()
        for array in inner_dict.values()
        for element in array
    })

    return flattened_words


def filter_words_map(words_map, guess_results):
    """
    Handle guess results in order to update possible words dict
    """
    if not guess_results:
        return words_map
    
    # Set params
    possible_words_sets = []

    # Count how many times the letter appears
    all_guess_results = guess_results['correct'] + guess_results['present'] + guess_results['incorrect']
    guess_letters = {result['letter'] for result in all_guess_results}
    letter_count = {
        letter: sum(
            1 for r in all_guess_results
            if r['letter'] == letter and r['status'] != 0
        )
        for letter in guess_letters
    }

    # Handle each letter result
    for result in guess_results['correct']:
        possible_words_sets.append(
            handle_correct_status(words_map, result)
        )
    for result in guess_results['incorrect']:
        possible_words_sets.append(
            handle_incorrect_status(words_map, result, letter_count)
        )
    for result in guess_results['present']:
        possible_words_sets.append(
            handle_present_status(words_map, result, letter_count)
        )

    # Get intersection of possible words and recreate mapping
    possible_words = list(set.intersection(*possible_words_sets))
    words_map = create_words_map(possible_words)

    return words_map


def handle_correct_status(words_map, result):
    """
    Letter is in the correct position
    """
    letter = result['letter'].lower()
    pos = result['pos']

    words_with_letter_pos = set(words_map[letter][pos])

    return words_with_letter_pos


def handle_present_status(words_map, result, letter_count):
    """
    Letter is in the word but NOT at this position
    """
    letter = result['letter'].lower()
    pos = result['pos']

    words_with_letter_elsewhere = {
        word
        for p, word_list in words_map[letter].items()
        if p != pos
        for word in word_list
        if word.count(letter) == letter_count[letter] and pos not in [i for i, c in enumerate(word) if c == letter]
    }

    return words_with_letter_elsewhere


def handle_incorrect_status(words_map, result, letter_count):
    """
    Letter appears in the word a defined amount of times
    """
    letter = result['letter'].lower()

    words_with_num_of_letters = {
        word
        for word in flatten_words_map(words_map)
        if word.count(letter) == letter_count[letter]
    }
    
    return words_with_num_of_letters