# UNCOMMENT THINGS WHEN CHANGE GUROBI TO PULP
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir))
from utils.instance_utils import *
from utils.solve_utils import *
from utils.wordle_tools_utils import *


def main(flags, game_results=[]):
    # Get instance
    instance = get_instance()
    all_words, num_of_letters, num_of_attempts, possible_words_dict = instance
    
    # Create model
    # model = create_model(instance)

    # Handle testing mode or normal mode
    if flags['test']:
        selected_word = get_random_word(all_words)

        for _ in range(num_of_attempts):
            word_guess = solve(model, game_results)
            print(word_guess)
            game_results = update_game_results(game_results, selected_word, word_guess)
            possible_words_dict = update_possible_words_dict(possible_words_dict, game_results)
    else:
        word_guess = get_random_word(all_words)
        # word_guess = solve(model, game_results)

    return word_guess


if __name__ == "__main__":
    flags = {
        "test": True,
        'flask': False
    }

    main(flags=flags)