from .instance_utils import *
from .calculation_utils import *
import gurobipy as gp
from gurobipy import GRB
import random


def solve_random(instance):
    """
    Solve model for random word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance

    # Solve
    word_guess = random.choice(key_words)

    return word_guess


def solve_greedy(instance):
    """
    Solve model for optimal word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance
    
    # Solve
    word_guess = max(key_words, key=lambda word: get_word_probability(word, key_words))

    return word_guess


def solve_trinary_search(instance):
    """
    Solve model for trinary search word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance

    # Distances
    


def presolve_diversification(instance):
    """
    Presolve model for diversification and optimal word guess
    """
    # Unpack data
    all_words, words, key_words, num_of_letters, num_of_attempts = instance
    alfabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    num_of_guesses = 3

    model = gp.Model("Max_Diversification")
    model.setParam('OutputFlag', 0)

    # Add decision variables
    x = model.addVars(key_words, vtype=GRB.BINARY, name="x")
    y = model.addVars(alfabet, vtype=GRB.BINARY, name="y")
    z = model.addVars(words, vtype=GRB.BINARY, name="z")
            
    # Set objective function
    obj_fn = gp.quicksum(x[w] for w in key_words)
    model.setObjective(obj_fn, GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(gp.quicksum(z[w] for w in words) == num_of_guesses, name="c1")
    model.addConstrs(
        (y[l] <= gp.quicksum(z[w] for w in words if l in w) for l in alfabet),
        name="c2"
    )
    model.addConstrs((z[w] <= x[w] for w in key_words), name="c3")
    model.addConstrs((y[l] <= x[w] for w in key_words for l in w), name="c4")
    model.addConstrs((x[w] <= gp.quicksum(y[l] for l in w) for w in key_words), name="c5")

    # Solve
    model.optimize()

    # Get solution
    guesses = [w for w in words if z[w].x > 0.5]

    return guesses