import random


def get_random_word(words):
    return random.choice(words)


def filter_words(words, guess_results):
    """
    Handle guess results in order to update possible words
    """
    if not guess_results:
        return words

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
        words = handle_correct_status(words, result)
    for result in guess_results['incorrect']:
        words = handle_incorrect_status(words, result, letter_count)
    for result in guess_results['present']:
        words = handle_present_status(words, result, letter_count)

    return words


def handle_correct_status(words, result):
    """
    Letter is in the correct position
    """
    letter = result['letter'].lower()
    pos = result['pos']

    words_with_letter_pos = {word for word in words if word[pos] == letter}

    return words_with_letter_pos


def handle_present_status(words, result, letter_count):
    """
    Letter is in the word but NOT at this position
    """
    letter = result['letter'].lower()
    pos = result['pos']

    words_with_letter_elsewhere = {
        word
        for word in words
        if p != pos
        for word in word_list
        if word.count(letter) == letter_count[letter] and pos not in [i for i, c in enumerate(word) if c == letter]
    }

    return words_with_letter_elsewhere


def handle_incorrect_status(words, result, letter_count):
    """
    Letter appears in the word a defined amount of times
    """
    pos = result['pos']
    letter = result['letter'].lower()

    words_with_num_of_letters = {
        word
        for word in words
        if word.count(letter) <= letter_count[letter]
        if word[pos] != letter
    }
    
    return words_with_num_of_letters
