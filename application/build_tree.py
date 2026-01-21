from classes.guess_tree import *
from classes.results import *
from utils.instance_utils import *


flags = {
    'print_diagnosis': True,
    'evaluate': True,
    'save_tree': False
}

configs = {
    'GPU': False,
    'hard_mode': True,
    'metric': 1       # 0 -> Avg. Size     1 -> Subtree-10     2 -> Subtree-Full
}


def main():
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