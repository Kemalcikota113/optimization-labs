import numpy as np
from numpy.random import rand, randn
from scipy.optimize import linprog, OptimizeResult
from time import perf_counter
import warnings # To suppress warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# example of a problem isntance

n, m = 10, 10
A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis = -1)
b = np.ones(m)
c = np.concatenate([rand(n), np.zeros(m)])

# Answer to i) --> Because A is an indentity matrix, means the feasible region is bounded.
# b ensures that constraints can be satisfied, making the feasible region non empty.

# Extract fractional seconds (high resolutions)
start_time = perf_counter()

result = linprog(-c, 
                 A_eq=A, 
                 b_eq=b, 
                 method='simplex', 
                 options={'maxiter': 5000})

elapsed_time = 1000*(perf_counter() - start_time)

print(f"Optimization result: {result}")
print(f"Elapsed time (ms): {elapsed_time}")

# answer to ii) --> 6-7 ms to run on home desktop