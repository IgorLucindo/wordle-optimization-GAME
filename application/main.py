# application/main.py
import asyncio
import js # Import the 'js' module to interact with the JavaScript global scope
import numpy as np
from scipy.optimize import milp, Bounds

async def run_scipy_mip():
    """
    Solves a MIP problem using the native SciPy MILP solver,
    which is a core part of the Pyodide scientific stack.
    """
    print("Python: SciPy is pre-loaded. Building model...")

    # --- Problem Definition ---
    # Minimize: 5*x1 + 8*x2 + 10*x3
    # Subject to: 2*x1 + 3*x2 + 4*x3 >= 5
    # Variables: x1, x2, x3 are binary (0 or 1).

    # 1. Define objective function coefficients
    c = np.array([5, 8, 10])

    # 2. Define the constraints.
    # The function requires constraints in the form A_ub @ x <= b_ub.
    # We must convert "2*x1 + 3*x2 + 4*x3 >= 5"
    # into         "-2*x1 - 3*x2 - 4*x3 <= -5"
    A_ub = np.array([[-2, -3, -4]])
    b_ub = np.array([-5])

    # 3. Define variable bounds and integrality.
    # All variables are binary (bounds 0 to 1) and are integers.
    # We use the scipy.optimize.Bounds object for robustness.
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones_like(c) # A 1 for each variable means it's an integer.

    # 4. Solve the problem
    print("Python: Problem defined. Solving with SciPy MILP...")
    res = milp(c=c, constraints=(A_ub, b_ub, None), integrality=integrality, bounds=bounds)

    print(f"Python: SciPy solver status: {res.message}")
    
    # 5. Process and store results.
    results = {}
    if res.success:
        results = {
            "status": "Optimal",
            "objective_value": res.fun,
            "x1": res.x[0],
            "x2": res.x[1],
            "x3": res.x[2]
        }
    else:
        results = {
            "status": res.message,
            "objective_value": None,
            "x1": None, "x2": None, "x3": None
        }
    
    print(f"Python: Storing results for JavaScript: {results}")

    # Set a variable on the JavaScript global scope (`window` or `globalThis`)
    # by assigning an attribute to the imported 'js' object.
    js.pyodide_results = results


# This ensures the async function is run when the script is executed
asyncio.ensure_future(run_scipy_mip())
