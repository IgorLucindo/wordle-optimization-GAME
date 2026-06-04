from classes.guess_tree import *
from classes.results import *
from classes.instance_loader import InstanceLoader
from utils.cli_parser import get_args, resolve_runtime_args, discover_available_instances


def run_instance(instance_name, flags, configs, dataset_dir):
    # Deep-copy configs so each run gets a clean slate
    run_configs = dict(configs)
    run_configs['data'] = instance_name
    run_configs['GPU'] = configs['GPU']

    print(f"\n{'='*50}\nRunning: {instance_name}\n{'='*50}")

    loader = InstanceLoader(
        instance_name=instance_name,
        use_gpu=run_configs['GPU'],
        dataset_dir=dataset_dir
    )
    instance = loader.get_full_instance(flags, run_configs)
    G_names, T_names = instance[0], instance[1]

    gt = Guess_Tree(instance, flags, run_configs)
    tree, runtime = gt.build_tree()

    results = Results(flags, run_configs)
    results.set_data(tree, runtime, n_G=len(G_names), n_T=len(T_names))
    results.evaluate()
    results.print()
    results.save()


def main():
    args = get_args()
    flags, configs, dataset_dir = resolve_runtime_args(args)

    if configs['data'] == 'all':
        discovered = discover_available_instances(dataset_dir)
        for instance_name in sorted(discovered):
            run_instance(instance_name, flags, configs, dataset_dir)
    else:
        run_instance(configs['data'], flags, configs, dataset_dir)


if __name__ == "__main__":
    main()
