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
print(f"Required price increase: {price_increase:.2f}\n\n")

# answer to iii) -- put all extra hours on stage1.

# answer to iv) ---> because the current price is already 1000, no increase is required.

# -----------------------------------

# Add Type C TV
profit_C = 1350
stage_times_C = [7, 4, 2]

# Compute reduced cost for Type C
reduced_cost_C = profit_C - (stage_times_C[0] * y1 + stage_times_C[1] * y2 + stage_times_C[2] * y3)
print(f"Reduced cost for Type C TV: {reduced_cost_C:.2f}")



# Decision: Should Type C TV be produced?
if -reduced_cost_C < 0:
    print("Type C TV should be produced.")
    # Update the primal problem
    c_primal = np.array([-700, -1000, -1350])  # Negate for maximization to minimization
    A_primal = np.array([
        [3, 5, 7],    # Stage I hours per unit
        [1, 3, 4],    # Stage II hours per unit
        [2, 2, 2]     # Stage III hours per unit
    ])
    b_primal = np.array([3900, 2100, 2200])  # Original constraints

    # Solve the updated primal problem

    result_primal = linprog(c_primal, A_ub=A_primal, b_ub=b_primal, bounds=(0, None), method='highs')
    print(result_primal)

    if result_primal.success:
        new_profit = -result_primal.fun  # Negate back to maximize
        production_plan = result_primal.x

        print(f"New optimum profit: {new_profit:,.2f}")
        print(f"New production plan: Type A = {production_plan[0]:.2f}, Type B = {production_plan[1]:.2f}, Type C = {production_plan[2]:.2f}\n\n")
    else:
        print("Re-optimization failed!")
else:
    print("Type C TV should NOT be produced.")

# answer to v) ---->
# the reduced cost is 50 which means that type C's profit 50 is higher than the contribution needed to maintain feasbility.
# a positive reduced cost for a max problem means that including this new variable would NOT increase the optimal profit.

# -----------------------------------

# Optimal production plan after including Type C
x1, x2, x3 = 950, 0, 150  # Current production levels

# Quality inspection times per unit
inspection_time_A = 0.5  # hours for Type A
inspection_time_B = 0.75  # hours for Type B
inspection_time_C = 0.1  # hours for Type C

# Compute total inspection time required
total_inspection_time = (inspection_time_A * x1) + (inspection_time_B * x2) + (inspection_time_C * x3)


# Output results
print(f"Total inspection time required: {total_inspection_time:.2f} hours")

