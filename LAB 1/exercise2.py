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
print(f"Elapsed time (ms): {elapsed_time}\n\n")

# answer to ii) --> 6-7 ms to run on home desktop

# this is the same code from before just in a function
def run_experiment_simplex(n):

    m = n  # Keep m = n
    elapsed_times = []

    for _ in range(5):  # Perform 5 trials
        # Generate problem instance
        A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis=-1)
        b = np.ones(m)
        c = np.concatenate([rand(n), np.zeros(m)])
        
        # Measure elapsed time
        start_time = perf_counter()
        linprog(-c, A_eq=A, b_eq=b, method='simplex', options={'maxiter': 5000})
        elapsed_time = 1000 * (perf_counter() - start_time)  # Convert to milliseconds
        elapsed_times.append(elapsed_time)
    
    # Return average elapsed time
    return np.mean(elapsed_times)

"""

results = {}
for n in [10, 20, 50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:  # Adjust values as needed
    avg_time = run_experiment_simplex(n)
    results[n] = avg_time
    print(f"Average time for n = m = {n}: {avg_time:.2f} ms")
"""
# answer to iii) --> I can run about 140-150 under 1 second depending on the run on home desktop.


def run_experiment_highs(n):

    m = n  # Keep m = n
    elapsed_times = []

    for _ in range(5):  # Perform 5 trials
        # Generate problem instance
        A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis=-1)
        b = np.ones(m)
        c = np.concatenate([rand(n), np.zeros(m)])
        
        # Measure elapsed time
        start_time = perf_counter()
        linprog(-c, A_eq=A, b_eq=b, method='highs', options={'maxiter': 5000})
        elapsed_time = 1000 * (perf_counter() - start_time)  # Convert to milliseconds
        elapsed_times.append(elapsed_time)
    
    # Return average elapsed time
    return np.mean(elapsed_times)

results = {}
for n in [100, 200, 500, 1000, 1100, 1200, 1300, 1400, 1500]:  # Adjust values as needed
    avg_time = run_experiment_highs(n)
    results[n] = avg_time
    print(f"Average time for n = m = {n}: {avg_time:.2f} ms")

# answer to iv) --> I can run about 1100 under 1 second depending on the run on home desktop.