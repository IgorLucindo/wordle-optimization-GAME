from classes.results import *
from utils.instance_utils import *
from utils.simulate_utils import *


flags = {
    'save_results': True,
    'solver_type': 'random'          # 'random', 'greedy', 'diver_grd' or 'guess_tree'
}


def main():
    results = Results(flags)
    instance = get_instance()

    # Simulate game with solver
    simulation_results = simulate_games(instance, flags['solver_type'])

    results.set_data(simulation_results)
    results.save()


if __name__ == "__main__":
    main()