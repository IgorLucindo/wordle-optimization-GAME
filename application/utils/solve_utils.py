from utils.wordle_tools_utils import *
from pulp import *
import random


def create_model(instance):
    """
    Create model for optimal word guess
    """
    # Unpack data
    words, num_of_letters, num_of_attempts, words_map = instance

    # Create model
    model = LpProblem("OptimalGuess", LpMaximize)

    # Add decision variables
    x = LpVariable("x", lowBound=0, cat=LpInteger)
    y = LpVariable("y", lowBound=0, cat=LpInteger)

    # Set objective function
    model += 20 * x + 30 * y, "objfn"

    # Add constraints
    model += 2 * x + 3 * y <= 120, "c1"
    model += 1 * x + 2 * y <= 80, "c2"
    model += x >= 10, "c3"

    return model


def solve(model, instance):
    """
    Solve model for optimal word guess
    """
    # Unpack data
    words, num_of_letters, num_of_attempts, words_map = instance

    # Solve
    model.solve(PULP_CBC_CMD(msg=0))


    # START TEST
    # Get random possible word
    possible_idx = flatten_words_map(words_map)
    word_guess = random.choice(list(possible_idx))
    # END TEST


    return word_guess


def simulate_game_solver(model, instance):
    """
    Simulate wordle and completely solve it for testing model
    """
    # Unpack data
    words, num_of_letters, num_of_attempts, words_map = instance
    game_results = []
    
    # Select random word
    selected_word = get_random_word(words)

    # Solve game
    for _ in range(num_of_attempts):
        word_guess = solve(model, instance)
        print(word_guess)
        game_results = update_game_results(game_results, selected_word, word_guess)
        words_map = filter_words_map(words_map, game_results)

    return word_guess