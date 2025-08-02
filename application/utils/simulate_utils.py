from .instance_utils import *
from .solve_utils import *
import time


def simulate_games(instance, solver_type):
    """
    Simulate wordle games and solve them for all words
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance
    solvers = {
        'random': solve_random,
        'greedy': solve_greedy,
        'guess_tree': solve_guess_tree
    }
    solver = solvers[solver_type]
    simulation_results = {'solver_type': solver_type, 'list': [], 'runtime': 0}

    # Solve games
    for target_word in key_words:
        start_time = time.time()
        simulation_results['list'].append(simulate_game_solver(instance, target_word, solver))
        simulation_results['runtime'] += time.time() - start_time

    return simulation_results


def simulate_game_solver(instance, target_word, solver):
    """
    Simulate wordle and completely solve it for testing model
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts, tree = instance

    word_guess = None
    results = {
        'guesses': [],
        'correct_guess': False
    }

    # Simulate
    for _ in range(num_of_attempts):
        feedback = get_feedback(target_word, word_guess)
        instance = fiter_instance(instance, word_guess, feedback)

        # Solve word guess
        word_guess = solver(instance, feedback)
        
        # Set results
        results['guesses'].append(word_guess)

        # Stop condition
        if word_guess == target_word:
            results['correct_guess'] = True
            break

    return results