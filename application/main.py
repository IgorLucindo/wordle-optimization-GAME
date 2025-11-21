from utils.instance_utils import *
from classes.guess_tree import *


flags = {
    'print_diagnosis': True,
    'evaluate': True,
    'save_tree': False,
    'save_results': False
}

configs = {
    'GPU': True,
    'composite_score': True,
    'hard_mode': False,
    '#trees': 10
}


def main():
    instance = get_instance(configs)

    gt = Guess_Tree(instance, flags, configs)

    # gt.start_diagnosis()
    # gt.build()
    # gt.stop_diagnosis()
    # gt.evaluate()
    # gt.print_results()
    # gt.save()

    gt.start_diagnosis()
    gt.build_for_all_words()
    gt.stop_diagnosis()
    

if __name__ == "__main__":
    main()