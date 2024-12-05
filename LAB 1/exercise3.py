import numpy as np
from scipy.optimize import linprog


c_dual = np.array([3900, 2100, 2200])  # Minimize w  --> this is also C
A_dual = np.array([
    [-3, -1, -2],  # Negate to convert >= to <=
    [-5, -3, -2]    # this is also A
])
b_dual = np.array([-700, -1000])  # this is b


result_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')

# Extract results
if result_dual.success:
    dual_min_value = result_dual.fun
    dual_variables = result_dual.x
    print("Dual Optimization successful!")
    print(f"Minimum cost: {dual_min_value:,.2f}")
    print(f"Optimal dual variables: y1 = {dual_variables[0]:.2f}, y2 = {dual_variables[1]:.2f}, y3 = {dual_variables[2]:.2f}")
else:
    print("Dual Optimization failed!")
    print(result_dual.message)

# this gives correct output of 860.000 as in exercise 1.py

# -----------------------------------