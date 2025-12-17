from utils.instance_utils import *
from classes.guess_tree import *


flags = {
    'print_diagnosis': False,
    'evaluate': True,
    'save_tree': False,
    'save_results': False
}

configs = {
    'GPU': False,
    'hard_mode': False,
    'subtree_score': False
}


def main():
    instance = get_instance(flags, configs)

    gt = Guess_Tree(instance, flags, configs)
    gt.load_tree('dataset/guess_tree.json')
    gt.evaluate_decoded()
    gt.print_results()
    

if __name__ == "__main__":
    main()