from utils.instance_utils import *
from utils.simulate_utils import *


flags = {
    'solver': 'random',     # 'random', 'optimal' or
    'save_results': True
}


def main():
    # Get instance
    instance = get_instance()

    # Simulate game with solver
    results = simulate_games(instance, flags)


if __name__ == "__main__":
    main()