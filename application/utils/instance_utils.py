from collections import defaultdict


def get_instance():
    all_words = _get_all_words()
    num_of_letters = len(all_words[0])
    possible_words_dict = _get_possible_words_dict(all_words)
    num_of_attempts = 6

    return all_words, num_of_letters, num_of_attempts, possible_words_dict


def _get_possible_words_dict(all_words):
    possible_words_dict = defaultdict(lambda: defaultdict(list))

    for word in all_words:
        for i in range(len(word)):
            possible_words_dict[word[i]][i].append(word)

    return possible_words_dict


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