from utils.instance_utils import *
from classes.guess_tree import *


flags = {
    'print_diagnosis': True,
    'evaluate': True,
    'save_tree': False
}

configs = {
    'GPU': False,
    'hard_mode': True,
    'subtree_score': False
}


def main():
    instance = get_instance(flags, configs)

    gt = Guess_Tree(instance, flags, configs)
    gt.start_diagnosis()
    gt.build_tree()
    gt.stop_diagnosis()
    gt.evaluate()
    gt.print_results()
    gt.save()


if __name__ == "__main__":
    main()