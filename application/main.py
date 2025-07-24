from classes.results import *
from utils.instance_utils import *
from utils.simulate_utils import *


flags = {
    'solver': 'random',     # 'random', 'optimal' or
    'save_results': True
}


def main():
    results = Results(flags)
    instance = get_instance()

    # Simulate game with solver
    simulation_results = simulate_games(instance, flags)

    results.set_data(simulation_results)
    results.save()


if __name__ == "__main__":
    main()