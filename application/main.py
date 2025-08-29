from utils.instance_utils import *
from classes.decision_tree import *


flags = {
    'print_diagnosis': True,
    'evaluate': True,
    'save_tree': False,
    'save_results': False
}


def main():
    instance = get_instance()

    dt = Decision_Tree(instance, flags)
    dt.start_diagnosis()
    dt.create()
    dt.stop_diagnosis()
    dt.evaluate()
    dt.save()
    

if __name__ == "__main__":
    main()