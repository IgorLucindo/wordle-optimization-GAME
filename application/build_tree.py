from classes.guess_tree import *
from classes.results import *
from classes.instance_loader import InstanceLoader
from utils.cli_parser import get_args, resolve_runtime_args


def main():
    args = get_args()
    flags, configs, dataset_dir = resolve_runtime_args(args)

    # Load instance using InstanceLoader
    loader = InstanceLoader(
        instance_name=configs['data'],
        use_gpu=configs['GPU'],
        dataset_dir=dataset_dir
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
