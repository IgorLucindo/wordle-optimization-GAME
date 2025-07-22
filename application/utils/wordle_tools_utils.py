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


def filter_words_map(words_map, game_results):
    """
    Handle game results in order to update possible words dict
    """
    # Set params
    possible_words_sets = []
    status_map = {'correct': 2, 'present': 1, 'incorrect': 0}
    num_of_letters = next(
        (len(word) for d in words_map.values() for lst in d.values() for word in lst),
        None
    )

    # Handle each letter result
    for result in game_results:
        possible_words_sets.append(
            handle_status(status_map, words_map, result, game_results, num_of_letters)
        )

    # Get intersection of possible words and recreate mapping
    possible_words = list(set.intersection(*possible_words_sets))
    words_map = create_words_map(possible_words)

    return words_map


def handle_status(status_map, words_map, result, game_results, num_of_letters):
    """
    Handle results status in order to update possible words set
    """
    # Count how many times the letter appears
    letter_count = sum(
        1 for r in game_results[-num_of_letters:]
        if r['letter'] == result['letter'] and r['status'] != 0
    )

    # Handle status
    if result['status'] == status_map['correct']:
        possible_words = handle_correct_status(words_map, result)
    elif result['status'] == status_map['present']:
        possible_words = handle_present_status(words_map, result, letter_count)
    elif result['status'] == status_map['incorrect']:
        possible_words = handle_incorrect_status(words_map, result, letter_count)

    return possible_words


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
        if word.count(letter) == letter_count and pos not in [i for i, c in enumerate(word) if c == letter]
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
        if word.count(letter) == letter_count
    }
    
    return words_with_num_of_letters
    

def update_game_results(game_results, selected_word, word_guess):
    """
    Update game results for a single guess
    """
    for i in range(len(selected_word)):
        if selected_word[i] == word_guess[i]:
            game_results.append({'letter': selected_word[i], 'pos': i, 'status': 2})
        elif selected_word[i] in word_guess:
            game_results.append({'letter': selected_word[i], 'pos': i, 'status': 1})
        else:
            game_results.append({'letter': selected_word[i], 'status': 0})

    return game_results