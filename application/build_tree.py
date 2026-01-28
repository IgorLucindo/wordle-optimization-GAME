from classes.guess_tree import *
from classes.results import *
from utils.instance_utils import *
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Build and optimize a Wordle decision tree.")
    
    # Configs
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (disable GPU)')
    parser.add_argument('--hard_mode', action='store_true', help='Enable Hard Mode constraints')
    parser.add_argument('--metric', type=int, default=1, choices=[0, 1, 2], help='Optimization metric: 0=Avg. Size, 1=Subtree-k, 2=Subtree-Full')
    parser.add_argument('--k', type=int, default=15, help='Top-k candidates to evaluate (for metric 1)')
    
    # Flags
    parser.add_argument('--no_diagnosis', action='store_true', help='Disable diagnosis printing')
    parser.add_argument('--no_evaluate', action='store_true', help='Skip evaluation step')
    parser.add_argument('--save_tree', action='store_true', help='Save the resulting tree to JSON')

    return parser.parse_args()


def main():
    args = get_args()

    # Get flags and configs from args
    flags = {
        'print_diagnosis': not args.no_diagnosis,
        'evaluate': not args.no_evaluate,
        'save_tree': args.save_tree
    }
    configs = {
        'GPU': not args.cpu,
        'hard_mode': args.hard_mode,
        'metric': args.metric,
        'k': args.k
    }

    instance = get_instance(flags, configs)

    gt = Guess_Tree(instance, flags, configs)
    tree, D, runtime = gt.build_tree()

    results = Results(instance, flags, configs)
    results.set_data(tree, runtime)
    results.evaluate()
    results.print()
    results.save()


if __name__ == "__main__":
    main()