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
        'guess_tree': {'solver': solve_guess_tree, 'presolve': presolve_guess_tree}
    }
    simulation_results = {'solver_type': solver_type, 'list': [], 'runtime': 0}
    sim_data = simulation_data[solver_type]

    # Solve games
    for target_word in key_words:
        start_time = time.time()
        simulation_results['list'].append(simulate_game_solver(instance, target_word, sim_data))
        simulation_results['runtime'] += time.time() - start_time

    return simulation_results


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
    presolve_solution = None

    # Presolve
    if sim_data.get('presolve'):
        presolve_solution = sim_data['presolve'](instance)

    # Simulate
    for _ in range(num_of_attempts):
        feedback = get_feedback(target_word, word_guess)
        instance = fiter_instance(instance, word_guess, feedback)

        # Solve word guess
        word_guess = sim_data['solver'](instance, presolve_solution, feedback)
        
        # Set results
        results['guesses'].append(word_guess)

        # Stop condition
        if word_guess == target_word:
            results['correct_guess'] = True
            break

    return results