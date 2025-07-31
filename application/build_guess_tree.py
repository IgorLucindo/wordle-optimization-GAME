from utils.instance_utils import *
from classes.guess_tree import *


def main():
    instance = get_instance()

    guess_tree = Guess_Tree(instance)
    guess_tree.build(guess_tree.key_words)
    guess_tree.save()
    

if __name__ == "__main__":
    main()