import argparse
from pathlib import Path


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


def discover_available_instances(dataset_dir='data'):
    """Return mapping {instance_name: dataset_dir_path} for folders containing rules.json."""
    candidates = [Path(dataset_dir), Path('..') / dataset_dir]
    discovered = {}

    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue

        for instance_dir in sorted(base.iterdir()):
            if instance_dir.is_dir() and (instance_dir / 'rules.json').exists():
                discovered.setdefault(instance_dir.name, str(base))

    return discovered


def prompt_for_instance(instance_names):
    """Prompt the user to choose an instance by index or name."""
    print("No --data provided. Please choose an instance:")
    for idx, name in enumerate(instance_names, start=1):
        print(f"  {idx}. {name}")
    print(f"  {len(instance_names) + 1}. all")

    while True:
        user_input = input("Enter number or instance name: ").strip()

        if user_input.isdigit():
            selected_idx = int(user_input)
            if selected_idx == len(instance_names) + 1:
                return 'all'
            if 1 <= selected_idx <= len(instance_names):
                return instance_names[selected_idx - 1]
        elif user_input == 'all':
            return 'all'
        elif user_input in instance_names:
            return user_input

        print("Invalid choice. Please enter a valid number or instance name.")


def resolve_instance_selection(instance_name_arg):
    """Resolve chosen instance(s) and dataset root for InstanceLoader.
    
    Returns (selected, dataset_dir) where selected is either a single name
    or the string 'all'.
    """
    discovered = discover_available_instances('data')
    available_instances = sorted(discovered)

    if not available_instances:
        raise FileNotFoundError(
            "No instances found. Expected folders like data/<instance_name>/rules.json"
        )

    selected_instance = instance_name_arg
    if selected_instance is None:
        selected_instance = prompt_for_instance(available_instances)

    if selected_instance == 'all':
        # Return the common dataset dir (all instances share the same base)
        dataset_dir = discovered[available_instances[0]]
        return 'all', dataset_dir

    if selected_instance not in discovered:
        options = ', '.join(available_instances)
        raise ValueError(
            f"Unknown instance '{selected_instance}'. Available instances: {options}"
        )

    return selected_instance, discovered[selected_instance]


def get_args():
    parser = argparse.ArgumentParser(
        description="Build and optimize a guessing-game decision tree."
    )

    # Configs
    parser.add_argument('--data', type=str, default=None,
                        help='Instance to solve (e.g., wordle, wordle_hard, mastermind, zoo). If omitted, you will be prompted.')
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


def resolve_runtime_args(args):
    """Resolve selected instance, dataset root, flags, and configs from parsed args."""
    selected_instance, dataset_dir = resolve_instance_selection(args.data)

    flags = {
        'print_diagnosis': not args.no_diagnosis,
        'evaluate': not args.no_evaluate,
        'save_tree': args.save_tree
    }
    configs = {
        'GPU': not args.cpu,
        'explicit_cpu': args.cpu,
        'data': selected_instance,
        'k': parse_k_value(args.k),
        'score': args.score
    }

    return flags, configs, dataset_dir