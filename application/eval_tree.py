from classes.results import *
from classes.instance_loader import InstanceLoader
from pathlib import Path


def find_all_trees(data_dir='data'):
    """
    Discover all decision_tree.json files in data/ subdirectories.

    Returns list of (instance_name, tree_path) tuples.
    """
    # Handle both running from root and from application/ directory
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path = Path('..') / data_dir
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data directory at {data_dir} or ../{data_dir}")

    trees = []

    for instance_dir in data_path.iterdir():
        if not instance_dir.is_dir():
            continue

        tree_file = instance_dir / 'decision_tree.json'
        if tree_file.exists():
            trees.append((instance_dir.name, tree_file))

    return sorted(trees)


def main():
    # Evaluation doesn't need GPU or diagnosis - just simulates games from saved trees
    flags = {
        'print_diagnosis': False,
        'evaluate': True,
        'save_tree': False
    }
    configs = {
        'GPU': False,  # CPU is sufficient for evaluation
        'metric': 0,
        'k': 15,
        'score': 'PC'
    }

    # Find all decision trees
    trees = find_all_trees()

    if not trees:
        print("No decision trees found in data/")
        return

    print(f"\nFound {len(trees)} decision tree(s):\n")

    # Evaluate each tree
    for instance_name, tree_path in trees:
        print(f"{'='*60}")
        print(f"Instance: {instance_name}")
        print(f"Tree: {tree_path}")
        print(f"{'='*60}")

        try:
            # Load instance
            configs['game'] = instance_name
            loader = InstanceLoader(
                instance_name=instance_name,
                use_gpu=configs['GPU']
            )
            instance = loader.get_full_instance(flags, configs)

            # Evaluate tree
            results = Results(instance, flags, configs)
            results.load_tree(str(tree_path))

            # Use appropriate evaluation method based on instance type
            if loader.is_target.any():
                # Has self-identifying targets (Wordle, Mastermind)
                results.evaluate_decoded()
            else:
                # No self-identifying targets (Zoo) - would need evaluate() on raw tree
                print("Note: This instance has no self-identifying targets.")
                print("Loaded tree cannot be fully evaluated without raw tree data.\n")
                continue

            results.print()

        except Exception as e:
            print(f"Error evaluating {instance_name}: {e}\n")


if __name__ == "__main__":
    main()
