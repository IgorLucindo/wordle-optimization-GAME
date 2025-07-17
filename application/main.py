from utils.load_utils import *
from utils.solve_utils import *
from utils.wordle_tools_utils import *


def main():
    selected_word = get_random_word()
    all_words = get_all_words()
    num_of_letters = len(all_words[0])
    possible_words_dict = get_possible_words_dict(all_words)
    num_of_attempts = 6

    game_results = []

    instance = num_of_letters, num_of_attempts, possible_words_dict


    model = create_model(instance)

    for _ in range(num_of_attempts):
        word_guess = solve(model, game_results)
        print(word_guess)
        game_results = update_game_results(game_results, selected_word, word_guess)
        possible_words_dict = update_possible_words_dict(possible_words_dict, game_results)



if __name__ == "__main__":
    main()