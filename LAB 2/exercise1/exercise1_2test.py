import numpy as np
from scipy.optimize import linprog
import warnings

def solve_min_cost_flow(nodes, arcs, supplies, demands):
    """
    Solve a minimum cost flow problem:
    - nodes: list of node names
    - arcs: list of tuples (from_node, to_node, cost, capacity)
    - supplies: dict {node: supply_amount} (positive for supply, 0 for transshipment)
    - demands: dict {node: demand_amount} (positive for demand)
      Note: demands can be integrated as negative supply to unify, but here we keep separate.
    """
    
    # Create index mapping for nodes and arcs
    node_index = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)
    num_arcs = len(arcs)
    
    # Cost vector and capacity bounds
    costs = []
    bounds = []
    for (u, v, c, cap) in arcs:
        costs.append(c)
        bounds.append((0, cap))
    
    # Construct A_eq and b_eq
    # Flow balance: (outflow - inflow) = supply - demand
    # For supply nodes: outflow - inflow = supply
    # For demand nodes: inflow - outflow = demand -> outflow - inflow = -demand
    # For intermediate nodes: outflow - inflow = 0
    
    A_eq = np.zeros((num_nodes, num_arcs))
    b_eq = np.zeros(num_nodes)
    
    # Set up supply/demand balance
    # supply > 0 means net outflow = supply
    # demand > 0 means net outflow = -demand
    # If node not in supplies or demands, it's 0.
    for n in nodes:
        sup = supplies.get(n, 0)
        dem = demands.get(n, 0)
        # outflow - inflow = supply - demand
        b_eq[node_index[n]] = sup - dem
    
    # Fill A_eq with arc incidence
    for j, (u, v, c, cap) in enumerate(arcs):
        A_eq[node_index[u], j] = 1   # outflow from u
        A_eq[node_index[v], j] = -1  # inflow to v

    # Solve
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    result = linprog(costs, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result


# -------------------------
# ORIGINAL SCENARIO (Given)
# -------------------------
nodes_orig = ["Almhult", "Markaryd", "Liatorp", "Osby", "Ljungby", "Alvesta", "Växjö"]
arcs_orig = [
    ("Almhult", "Liatorp", 18, 1000),
    ("Almhult", "Osby", 24, 1000),
    ("Almhult", "Växjö", 62, 1000),
    ("Osby", "Markaryd", 29, 1000),
    ("Liatorp", "Ljungby", 31, 500),
    ("Liatorp", "Växjö", 46, 1000),
    ("Markaryd", "Ljungby", 51, 2000),
    ("Ljungby", "Alvesta", 42, 500),
    ("Alvesta", "Växjö", 20, 2000),
]

supplies_orig = {"Almhult": 2500, "Markaryd": 1000}
demands_orig = {"Ljungby": 1000, "Alvesta": 500, "Växjö": 2000}

res_orig = solve_min_cost_flow(nodes_orig, arcs_orig, supplies_orig, demands_orig)
print("Original Scenario:")
print("Status:", res_orig.message)
print("Optimal cost:", res_orig.fun)
print("Flows:", res_orig.x)
print()


# -------------------------
# VÄRNAMO SCENARIO
# -------------------------
nodes_varnamo = ["Almhult", "Markaryd", "Liatorp", "Osby", "Ljungby", "Alvesta", "Växjö", "Värnamo"]

# Same original arcs plus 2 new arcs from Värnamo
arcs_varnamo = [
    ("Almhult", "Liatorp", 18, 1000),
    ("Almhult", "Osby", 24, 1000),
    ("Almhult", "Växjö", 62, 1000),
    ("Osby", "Markaryd", 29, 1000),
    ("Liatorp", "Ljungby", 31, 500),
    ("Liatorp", "Växjö", 46, 1000),
    ("Markaryd", "Ljungby", 51, 2000),
    ("Ljungby", "Alvesta", 42, 500),
    ("Alvesta", "Växjö", 20, 2000),
    ("Värnamo", "Ljungby", 43, 2000),
    ("Värnamo", "Alvesta", 50, 500),
]

supplies_varnamo = {"Almhult": 2000, "Markaryd": 750, "Värnamo": 750}
demands_varnamo = {"Ljungby": 1000, "Alvesta": 500, "Växjö": 2000}

res_varnamo = solve_min_cost_flow(nodes_varnamo, arcs_varnamo, supplies_varnamo, demands_varnamo)
print("Värnamo Scenario:")
print("Status:", res_varnamo.message)
print("Optimal cost:", res_varnamo.fun)
print("Flows:", res_varnamo.x)
print()


# -------------------------
# VISLANDA SCENARIO
# -------------------------
nodes_vislanda = ["Almhult", "Markaryd", "Liatorp", "Osby", "Ljungby", "Alvesta", "Växjö", "Vislanda"]

arcs_vislanda = [
    ("Almhult", "Liatorp", 18, 1000),
    ("Almhult", "Osby", 24, 1000),
    ("Almhult", "Växjö", 62, 1000),
    ("Osby", "Markaryd", 29, 1000),
    ("Liatorp", "Ljungby", 31, 500),
    ("Liatorp", "Växjö", 46, 1000),
    ("Markaryd", "Ljungby", 51, 2000),
    ("Ljungby", "Alvesta", 42, 500),
    ("Alvesta", "Växjö", 20, 2000),
    ("Vislanda", "Alvesta", 15, 500),
    ("Vislanda", "Växjö", 29, 500),
]

supplies_vislanda = {"Almhult": 2120, "Markaryd": 1000, "Vislanda": 380}
demands_vislanda = {"Ljungby": 1000, "Alvesta": 500, "Växjö": 2000}

res_vislanda = solve_min_cost_flow(nodes_vislanda, arcs_vislanda, supplies_vislanda, demands_vislanda)
print("Vislanda Scenario:")
print("Status:", res_vislanda.message)
print("Optimal cost:", res_vislanda.fun)
print("Flows:", res_vislanda.x)
print()

# Compare the two new scenarios
if res_varnamo.success and res_vislanda.success:
    if res_varnamo.fun < res_vislanda.fun:
        print("Värnamo facility offers the lower cost:", res_varnamo.fun)
    else:
        print("Vislanda facility offers the lower cost:", res_vislanda.fun)
else:
    print("One of the scenarios did not find a feasible solution.")



    