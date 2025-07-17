def get_all_words():
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