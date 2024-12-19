import numpy as np
from scipy.optimize import linprog
import warnings

# Cost vector (in the order of variables defined above)
costs = [18, 24, 62, 29, 31, 46, 51, 42, 20]

# Capacity constraints for each arc
capacity = [1000, 1000, 1000, 1000, 500, 1000, 2000, 500, 2000]

# Number of variables (arcs)
num_vars = len(costs)

# We have 7 nodes: Almhult, Markaryd, Liatorp, Osby, Ljungby, Alvesta, Växjö
# So we need a 7x9 matrix for A_eq
A_eq = np.zeros((7, num_vars))

# Define indices for clarity:
A_L = 0  # Almhult→Liatorp
A_O = 1  # Almhult→Osby
A_V = 2  # Almhult→Växjö
O_M = 3  # Osby→Markaryd
L_J = 4  # Liatorp→Ljungby
L_V = 5  # Liatorp→Växjö
M_J = 6  # Markaryd→Ljungby
J_Av = 7 # Ljungby→Alvesta
Av_V = 8 # Alvesta→Växjö

# Node constraints:

# Almhult (supply=2500): outflow - inflow = 2500
# Outflow: A_L, A_O, A_V; no inflow
A_eq[0, A_L] = 1
A_eq[0, A_O] = 1
A_eq[0, A_V] = 1

# Markaryd (supply=1000): outflow - inflow = 1000
# Outflow: M_J; Inflow: O_M
A_eq[1, M_J] = 1
A_eq[1, O_M] = -1

# Liatorp (0): outflow - inflow = 0
# Outflow: L_J, L_V; Inflow: A_L
A_eq[2, L_J] = 1
A_eq[2, L_V] = 1
A_eq[2, A_L] = -1

# Osby (0): outflow - inflow = 0
# Outflow: O_M; Inflow: A_O
A_eq[3, O_M] = 1
A_eq[3, A_O] = -1

# Ljungby (demand=1000): inflow - outflow = 1000
# Inflow: L_J, M_J; Outflow: J_Av
# Rewrite as: (L_J + M_J - J_Av) = 1000
A_eq[4, L_J] = 1
A_eq[4, M_J] = 1
A_eq[4, J_Av] = -1

# Alvesta (demand=500): inflow - outflow = 500
# Inflow: J_Av; Outflow: Av_V
# (J_Av - Av_V) = 500
A_eq[5, J_Av] = 1
A_eq[5, Av_V] = -1

# Växjö (demand=2000): inflow - outflow = 2000
# Inflow: A_V, L_V, Av_V; no outflow
# (A_V + L_V + Av_V) = 2000
A_eq[6, A_V] = 1
A_eq[6, L_V] = 1
A_eq[6, Av_V] = 1

b_eq = [2500, 1000, 0, 0, 1000, 500, 2000]

bounds = [(0, cap) for cap in capacity]

warnings.filterwarnings("ignore", category=DeprecationWarning)
result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print("Status:", result.message)
print("Optimal cost:", result.fun)
print("Flows:", result.x)
