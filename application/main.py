import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir))
from utils.instance_utils import *
from utils.solve_utils import *
from utils.wordle_tools_utils import *


def main():
    # Get instance
    instance = get_instance()
    
    # Create model
    model = create_model(instance)

    # Simulate game with solver
    word_guess = simulate_game_solver(model, instance)

    return word_guess


if __name__ == "__main__":
    main()