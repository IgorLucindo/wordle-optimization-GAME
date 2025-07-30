from .instance_utils import *
from .solve_utils import *
from itertools import product
import time


def simulate_games(instance):
    """
    Simulate wordle games and solve them for all words
    """
    # Unpack data
    words, key_words, num_of_letters, num_of_attempts = instance
    simulation_data = {
        'random': {'list': [], 'runtime': 0, 'solver': solve_random},
        'optimal': {'list': [], 'runtime': 0, 'solver': solve_optimal},
        'diver_opt': {'list': [], 'runtime': 0, 'solver': solve_optimal, 'presolve': presolve_diversification}
    }

    # Solve games
    for target_word, (sim_type, sim_data) in product(key_words, simulation_data.items()):
        start_time = time.time()
        sim_data['list'].append(simulate_game_solver(instance, target_word, sim_data))
        sim_data['runtime'] += time.time() - start_time

    return simulation_data


def simulate_game_solver(instance, target_word, sim_data):
    """
    Simulate wordle and completely solve it for testing model
    """
    # Unpack data
    words, key_words, num_of_letters, num_of_attempts = instance

    word_guess = None
    results = {
        'guesses': [],
        'correct_guess': False
    }
    first_guesses = []

    # Presolve
    if sim_data.get('presolve'):
        first_guesses = sim_data['presolve'](instance)

    # Simulate
    for _ in range(num_of_attempts):
        guess_results = get_guess_results(target_word, word_guess)
        instance = fiter_instance(instance, guess_results)

        # Solve word guess
        if first_guesses:
            word_guess = first_guesses.pop(0)
        else:
            word_guess = sim_data['solver'](instance)
        
        # Set results
        results['guesses'].append(word_guess)

        # Stop condition
        if word_guess == target_word:
            results['correct_guess'] = True
            break

    return results