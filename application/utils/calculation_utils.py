import cupy as cp


def get_letter_pos_probabiliy(letter, pos, words):
    """
    Return probability of letter being in a specific position
    """
    words_len = len(words)
    words_letter_pos_len = len([word for word in words if word[pos] == letter])

    return words_letter_pos_len / words_len


def get_word_probability(word, words):
    """
    Return probability of word based on probability of letters in word
    """
    word_probability = 1

    for pos, letter in enumerate(word):
        word_probability *= get_letter_pos_probabiliy(letter, pos, words)

    return word_probability


def encode_word(word):
    return cp.array([ord(c) - 97 for c in word], dtype=cp.int8)