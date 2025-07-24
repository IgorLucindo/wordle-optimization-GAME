from .instance_utils import *
from .solve_utils import *


def simulate_games(instance, flags):
    """
    Simulate wordle games and solve them for all words
    """
    # Unpack data
    words, num_of_letters, num_of_attempts = instance
    simulation_results = []

    # Solve games
    for selected_word in words:
        simulation_results.append(simulate_game_solver(instance, selected_word, flags))
        break

    return simulation_results


def simulate_game_solver(instance, selected_word, flags):
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
        guess_results = get_guess_results(selected_word, word_guess)
        instance = fiter_instance(instance, guess_results)

        # Solve
        if flags['solver'] == 'random':
            word_guess = solve_random(instance)
        
        # Set results
        results['guesses'].append(word_guess)

        # Stop condition
        if word_guess == selected_word:
            results['correct_guess'] = True
            break

    return results