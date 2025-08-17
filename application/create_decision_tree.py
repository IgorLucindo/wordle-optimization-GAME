from utils.instance_utils import *
from classes.decision_tree2 import *


flags = {
    'print_diagnosis': True
}


def main():
    instance = get_instance()

    decision_tree = Decision_Tree(instance, flags)
    decision_tree.create()
    decision_tree.save()
    

if __name__ == "__main__":
    main()