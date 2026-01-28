from classes.results import *
from utils.instance_utils import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved Wordle decision tree.")
    
    # Configs
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (disable GPU)')
    parser.add_argument('--hard_mode', action='store_true', help='Evaluate in Hard Mode')

    # Flags
    parser.add_argument('--no_diagnosis', action='store_true', help='Disable diagnosis printing')

    return parser.parse_args()


def main():
    args = get_args()

    # Get flags and configs from args
    flags = {
        'print_diagnosis': not args.no_diagnosis,
        'evaluate': True,
        'save_tree': False
    }
    configs = {
        'GPU': not args.cpu,
        'hard_mode': args.hard_mode,
        'metric': 0,
        'k': 15
    }

    filename = "decision_tree_hard.json" if configs['hard_mode'] else "decision_tree.json"
    filepath = 'dataset/' + filename

    instance = get_instance(flags, configs)

    results = Results(instance, flags, configs)
    results.load_tree(filepath)
    results.evaluate_decoded()
    results.print()


if __name__ == "__main__":
    main()