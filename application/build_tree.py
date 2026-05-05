from classes.guess_tree import *
from classes.results import *
from classes.instance_loader import InstanceLoader
import argparse


def parse_k_value(k_str):
    if k_str.lower() == 'full':
        return -1  # Sentinel value for subtree-full
    
    try:
        k_value = int(k_str)
        if k_value < 1:
            raise ValueError("k must be at least 1")
        return k_value
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"k must be a positive integer or 'full', got '{k_str}'")
        raise


def get_args():
    parser = argparse.ArgumentParser(
        description="Build and optimize a guessing-game decision tree."
    )

    # Configs
    parser.add_argument('--game', type=str, default='wordle',
                        help='Game instance to solve (e.g., wordle, wordle_hard, mastermind, zoo)')
    parser.add_argument('--cpu', action='store_true', help='Run on CPU only (disable GPU)')
    parser.add_argument('--k', type=str, default='15',
                        help='Number of candidates to evaluate: 1=greedy, N=subtree-N, full=subtree-full (default: 15)')
    parser.add_argument('--score', type=str, default='PC', choices=['PC', 'WA', 'H'],
                        help='Score rule: PC=Partition Count, WA=Weighted Average, H=Entropy (default: PC)')

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
        'game': args.game,
        'k': parse_k_value(args.k),
        'score': args.score
    }

    # Load instance using InstanceLoader
    loader = InstanceLoader(
        instance_name=configs['game'],
        use_gpu=configs['GPU']
    )
    instance = loader.get_full_instance(flags, configs)

    gt = Guess_Tree(instance, flags, configs)
    tree, runtime = gt.build_tree()

    results = Results(flags, configs)
    results.set_data(tree, runtime)
    results.evaluate()
    results.print()
    results.save()


if __name__ == "__main__":
    main()
