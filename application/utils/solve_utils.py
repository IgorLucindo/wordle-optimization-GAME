from .instance_utils import *
import random
# from pulp import *


# def create_model(instance):
#     """
#     Create model for optimal word guess
#     """
#     # Unpack data
#     words, num_of_letters, num_of_attempts, words_map = instance

#     # Create model
#     model = LpProblem("OptimalGuess", LpMaximize)

#     # Add decision variables
#     x = LpVariable("x", lowBound=0, cat=LpInteger)
#     y = LpVariable("y", lowBound=0, cat=LpInteger)

#     # Set objective function
#     model += 20 * x + 30 * y, "objfn"

#     # Add constraints
#     model += 2 * x + 3 * y <= 120, "c1"
#     model += 1 * x + 2 * y <= 80, "c2"
#     model += x >= 10, "c3"

#     return model


# def solve(instance):
#     """
#     Solve model for optimal word guess
#     """
#     # Unpack data
#     words, num_of_letters, num_of_attempts, words_map = instance

#     # Solve
#     model.solve(PULP_CBC_CMD(msg=0))


#     return word_guess


def solve_random(instance):
    """
    Solve model for random word guess
    """
    # Unpack data
    words, num_of_letters, num_of_attempts, words_map = instance

    # Solve
    possible_idx = flatten_words_map(words_map)
    word_guess = random.choice(list(possible_idx))

    return word_guess