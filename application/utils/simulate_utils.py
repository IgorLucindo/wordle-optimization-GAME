from .instance_utils import *
from .solve_utils import *
from itertools import product
import time


def simulate_games(instance):
    """
    Simulate wordle games and solve them for all words
    """
    # Unpack data
    words, num_of_letters, num_of_attempts = instance
    simulation_results = {
        'random': {'list': [], 'runtime': 0},
        'optimal': {'list': [], 'runtime': 0}
    }

    # Solve games
    for target_word, (solver_type, results) in product(words, simulation_results.items()):
        start_time = time.time()
        results['list'].append(simulate_game_solver(instance, target_word, solver_type))
        results['runtime'] += time.time() - start_time


    return simulation_results


def simulate_game_solver(instance, target_word, solver_type):
    """
    Simulate wordle and completely solve it for testing model
    """
    # Unpack data
    words, num_of_letters, num_of_attempts = instance

    word_guess = None
    results = {
        'guesses': [],
        'correct_guess': False
    }

    # Simulate
    for _ in range(num_of_attempts):
        guess_results = get_guess_results(target_word, word_guess)
        instance = fiter_instance(instance, guess_results)

        # Solve
        if solver_type == 'random':
            word_guess = solve_random(instance)
        elif solver_type == 'optimal':
            word_guess = solve_optimal(instance)
        
        # Set results
        results['guesses'].append(word_guess)

        # Stop condition
        if word_guess == target_word:
            results['correct_guess'] = True
            break

    return results