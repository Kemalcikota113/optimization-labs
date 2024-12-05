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

dual_min_value = result_dual.fun
dual_variables = result_dual.x
print("Dual Optimization successful!")
print(f"Minimum cost: {dual_min_value:,.2f}\n\n")
#print(f"Optimal dual variables: y1 () = {dual_variables[0]:.2f}, y2 = {dual_variables[1]:.2f}, y3 = {dual_variables[2]:.2f}\n\n")

# answer to i) ---> this gives correct output of 860.000 as in exercise 1.py

# -----------------------------------

# Output shadow prices
print("Shadow prices (dual variables):")
print(f"Stage I constraint: {dual_variables[0]:.2f} (y1)")
print(f"Stage II constraint: {dual_variables[1]:.2f} (y2)")
print(f"Stage III constraint: {dual_variables[2]:.2f} (y3)\n\n")

# answer to ii) ---> the shadow prices are: y1 = 150, y2 = 0, y3 = 125

# if we increase the 3900 to 3900, then the minimum cost will increase by 150.

# -----------------------------------

"""

# Adding 100 hours to each stage and computing the impact
print("Impact of adding 100 hours to each stage:")
extra_hours = 100

for i, stage in enumerate(["Stage I", "Stage II", "Stage III"]):
    # Modify the RHS for the current stage
    b_modified = c_dual.copy()
    b_modified[i] += extra_hours

    # Solve the modified dual problem
    result_modified = linprog(b_modified, A_ub=A_dual, b_ub=b_dual, bounds=(0, None), method='highs')

    if result_modified.success:
        new_min_cost = result_modified.fun
        cost_increase = new_min_cost - dual_min_value
        print(f"{stage}: Minimum cost increase = {cost_increase:,.2f}")
    else:
        print(f"{stage}: Solver failed")
"""
# answer to iii) ---> because the shadow price for y1 is the highest, we should dedicate as many hours as we can to stage I.
# If any hours are left, we dedicate them to stage III. Stage II is not worth it because the shadow price is 0.

# -----------------------------------


# Shadow prices
y1, y2, y3 = dual_variables  # From previous dual solution

# Current profit per unit for Type B TVs
current_c2 = 1000

# Compute new c2 value based on shadow prices
new_c2 = 5 * y1 + 3 * y2 + 2 * y3

# Calculate required price increase
price_increase = new_c2 - current_c2

print(f"New price per unit for Type B TVs: {new_c2:.2f}")
print(f"Required price increase: {price_increase:.2f}")

# answer to iv) ---> because the current price is already 1000, no increase is required.

# -----------------------------------

