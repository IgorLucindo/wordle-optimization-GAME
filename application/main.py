import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir))
from utils.instance_utils import *
from utils.solve_utils import *
from utils.wordle_tools_utils import *


def main(flags, game_results=[]):
    # Get instance
    instance = get_instance(game_results)
    
    # Create model
    model = create_model(instance)

    # Handle testing mode or normal mode
    if flags['test']:
        word_guess = simulate_game_solver(model, instance)
    else:
        word_guess = solve(model, instance)

    return word_guess


if __name__ == "__main__":
    flags = {
        "test": True,
        'flask': False
    }

    main(flags=flags)