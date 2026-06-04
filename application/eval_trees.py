from classes.instance_loader import InstanceLoader
from pathlib import Path
import csv
import numpy as np


def find_all_trees(data_dir='data'):
    """
    Discover all decision_tree.json files in data/ subdirectories.

    Returns list of (instance_name, tree_path) tuples.
    """
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


def evaluate_tree(tree):
    """Compute stats from a loaded tree dict. Returns a dict of metrics."""
    D = np.array([
        depth
        for _, _, is_terminal, depth in tree['vertices']
        if is_terminal
    ])

    n_vertices = len(tree['vertices'])
    metadata = tree['metadata']

    return {
        'n_G':         metadata['n_G'],
        'n_T':         metadata['n_T'],
        'n_vertices':  n_vertices,
        'exp_guesses': round(float(D.mean()), 3),
        'std_guesses': round(float(D.std()), 3),
        'max_guesses': int(D.max()),
        'score_rule':  metadata['score_rule'],
        'cpu_runtime': metadata.get('cpu_runtime', ''),
        'gpu_runtime': metadata.get('gpu_runtime', ''),
    }


def main():
    trees = find_all_trees()

    if not trees:
        print("No decision trees found in data/")
        return

    # Resolve output path relative to cwd or parent (handles running from application/)
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir = Path('..') / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / 'eval_results.csv'

    columns = [
        'instance', '|G|', '|T|', '|V|',
        'exp_queries', 'std_queries', 'max_queries',
        'score_rule', 'cpu_runtime_s', 'gpu_runtime_s',
    ]

    rows = []
    for instance_name, tree_path in trees:
        tree = InstanceLoader.load_tree(str(tree_path))
        stats = evaluate_tree(tree)
        rows.append({
            'instance':       instance_name,
            '|G|':            stats['n_G'],
            '|T|':            stats['n_T'],
            '|V|':            stats['n_vertices'],
            'exp_queries':    stats['exp_guesses'],
            'std_queries':    stats['std_guesses'],
            'max_queries':    stats['max_guesses'],
            'score_rule':     stats['score_rule'],
            'cpu_runtime_s':  stats['cpu_runtime'],
            'gpu_runtime_s':  stats['gpu_runtime'],
        })

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} row(s) to {out_path}")


if __name__ == "__main__":
    main()
