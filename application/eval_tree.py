from utils.instance_utils import *
from classes.results import *


flags = {
    'print_diagnosis': False,
    'evaluate': True,
    'save_tree': False
}

configs = {
    'GPU': True,
    'hard_mode': False,
    'metric': 0       # 0 -> Avg. Size     1 -> Subtree-10     2 -> Subtree-Full
}


def main():
    filename = "decision_tree_hard.json" if configs['hard_mode'] else "decision_tree.json"
    filepath = 'dataset/' + filename

    instance = get_instance(flags, configs)

    results = Results(instance, flags, configs)
    results.load_tree(filepath)
    results.evaluate_decoded()
    results.print()


if __name__ == "__main__":
    main()