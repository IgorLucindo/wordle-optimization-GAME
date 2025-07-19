# import gurobipy as gp
# from gurobipy import GRB


def create_model(instance):
    """
    Create model for maximum mean return
    """
    # Unpack data
    all_words, num_of_letters, num_of_attempts, possible_words_dict = instance

    # Create model
    model = gp.Model("Optimal_Guess")
    model.setParam('OutputFlag', 0)

    # Add decision variables
    x = model.addVars(V, vtype=GRB.CONTINUOUS, name="x")

    # Set objective function
    obj_fn = gp.quicksum(x[i] for i in V)
    model.setObjective(obj_fn, GRB.MAXIMIZE)

    # Add constraints
    model.addConstrs((x[i] <= y[i] for i in V), name="c6")

    # Update model
    model.update()

    return model


def solve(model):
    model.optimize()

    return 