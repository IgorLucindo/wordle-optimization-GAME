from utils.instance_utils import *
from classes.guess_tree import *


flags = {
    'print_diagnosis': False,
    'evaluate': True,
    'save_tree': False
}

configs = {
    'GPU': True,
    'hard_mode': True,
    'subtree_score': False
}


def main():
    filename = "decision_tree_hard.json" if configs['hard_mode'] else "decision_tree.json"

    instance = get_instance(flags, configs)

    gt = Guess_Tree(instance, flags, configs)
    gt.load_tree('dataset/' + filename)
    gt.evaluate_decoded()
    gt.print_results()


if __name__ == "__main__":
    main()