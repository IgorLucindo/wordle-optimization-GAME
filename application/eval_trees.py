from classes.results import Results
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
    # Minimal flags/configs - no instance needed!
    flags = {
        'print_diagnosis': False,
        'evaluate': True,
        'save_tree': False
    }
    configs = {}

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

        configs['data'] = instance_name

        # Load tree using InstanceLoader
        tree = InstanceLoader.load_tree(str(tree_path))

        # Evaluate tree (no instance needed!)
        results = Results(flags, configs)
        results.set_data(tree, runtime=0)
        results.evaluate()
        results.print()


if __name__ == "__main__":
    main()
