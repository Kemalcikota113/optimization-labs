import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import linprog


# Decision variables:
# x1 = number of type A TVs to produce
# x2 = number of type B TVs to produce

# Objective function coefficients (profits)
c = np.array([700, 1000])  # profit per unit for type A and B

# Constraint coefficients matrix A
# Row 1: Stage I time constraints (3(x1) + 5(x2) <= 3900)
# Row 2: Stage II time constraints (1(x1) + 3(x2) <= 2100)
# Row 3: Stage III time constraints (2(x1) + 2(x2) <= 2200)
A = np.array([
    [3, 5],    # Stage I hours per unit
    [1, 3],    # Stage II hours per unit
    [2, 2]     # Stage III hours per unit
])

# Right-hand side constraints
b = np.array([3900, 2100, 2200])

# Print the matrices to verify
print("Objective coefficients (c):")
print(c)
print("\nConstraint coefficients (A):")
print(A)
print("\nRight-hand side values (b):")
print(b)

# Lines space provided by Lab instructions
x = np.linspace(0, 1200, 100)

# Calculate y (x2) values for each constraint line
# y = (b - a1*x)/a2 for each constraint
y1 = (b[0] - A[0, 0]*x) / A[0, 1]  # Stage I
y2 = (b[1] - A[1, 0]*x) / A[1, 1]  # Stage II
y3 = (b[2] - A[2, 0]*x) / A[2, 1]  # Stage III

# Values given by observing the graph
vertices = np.array([
    [1100, 0],
    [300, 600],
    [0, 700],
    [800, 300],
    [0, 0],
])


# Testing all the Vertices to confirm the maximum value
print("\nCalculating values at each vertex (z = 700x₁ + 1000x₂):")
print("-" * 50)
max_val = float('-inf')
best_vertex = None

for v in vertices:
    z = c @ v  # crazy that this works
    print(f"At point ({v[0]}, {v[1]}): z = {z:,}")
    if z > max_val:
        max_val = z
        best_vertex = v

print("-" * 50)
print(
    f"Maximum occurs at ({best_vertex[0]}, {best_vertex[1]}) with value: {max_val:,}")


C = np.array([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5])


def calcLevelCurve(x, C):
    # this is like solving for X_2 ???
    return (C - c[0]*x) / c[1]


hull = ConvexHull(vertices)
plt.plot(vertices[:, 0], vertices[:, 1], 'o')

for simplex in hull.simplices:
    plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')


for k, C_val in enumerate(C, 1):
    y = (C_val - c[0]*x) / c[1]  # Calculate x2 based on the level curve equation
    z_max = 1140000
    y_max = (z_max - c[0]*x) / c[1]
    plt.plot(x, y_max, 'g--', label='Max z = 1,140,000') # support line

    plt.plot(x, y, 'r', alpha=k/8, label=f'z = {C_val:,.0f}') # level curves --> code from lab


# # Create the plot
plt.figure(figsize=(10, 8))
plt.plot(x, y1, label='Stage I: 3x₁ + 5x₂ = 3900')
plt.plot(x, y2, label='Stage II: x₁ + 3x₂ = 2100')
plt.plot(x, y3, label='Stage III: 2x₁ + 2x₂ = 2200')

# # Set axis limits
# plt.xlim(0, 1200)
# plt.ylim(0, 1200)

# # Add labels and title
# plt.xlabel('x₁ (Type A TVs)')
# plt.ylabel('x₂ (Type B TVs)')
# plt.title('Feasible Region for TV Production')

# # Add grid and legend
# plt.grid(True)
# plt.legend()




# Show the plot
plt.show()



# -----------------------------------
# Linear Programming Optimization (task vi and vii)

# Objective function coefficients (profit, negate for maximization)
c_minimize = -c  # Negate the coefficients to turn max into min

# Solve the linear program
result = linprog(c_minimize, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

# Extract the results
if result.success:
    max_profit = -result.fun  # Negate back to get the maximum profit
    optimal_solution = result.x
    print("Optimization successful!")
    print(f"Maximum profit: {max_profit:,.2f}")
    print(f"Optimal production levels: Type A TVs = {optimal_solution[0]:.2f}, Type B TVs = {optimal_solution[1]:.2f}")
else:
    print("Optimization failed!")
    print(result.message)


