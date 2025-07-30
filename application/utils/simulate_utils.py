from .instance_utils import *
from .solve_utils import *
import time


def simulate_games(instance, solver_type):
    """
    Simulate wordle games and solve them for all words
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance
    simulation_data = {
        'random': {'solver': solve_random},
        'greedy': {'solver': solve_greedy},
        'diver_grd': {'solver': solve_greedy, 'presolve': presolve_diversification},
        'trinary_search': {'solver': solve_trinary_search}
    }
    sim_data = simulation_data[solver_type]
    sim_data['list'] = []
    sim_data['runtime'] = 0

    # Solve games
    for target_word in key_words:
        start_time = time.time()
        sim_data['list'].append(simulate_game_solver(instance, target_word, sim_data))
        sim_data['runtime'] += time.time() - start_time

    return simulation_data


def simulate_game_solver(instance, target_word, sim_data):
    """
    Simulate wordle and completely solve it for testing model
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance

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