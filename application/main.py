from classes.results import *
from utils.instance_utils import *
from utils.simulate_utils import *


flags = {
    'save_results': True
}

config = {
    'solver': 'random',     # 'random', 'optimal' or
}


def main():
    results = Results(flags, config)
    instance = get_instance()

    # Simulate game with solver
    simulation_results = simulate_games(instance, config)

    results.set_data(simulation_results)
    results.save()


if __name__ == "__main__":
    main()