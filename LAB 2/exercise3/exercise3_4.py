import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    # Rosenbrock function f(x,y) = (1 - x)^2 + 100(y - x^2)^2
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Starting point
x0 = [-1, 1.5]

res = minimize(rosenbrock, x0)

print("Optimization result:", res)

final_gradient = res.jac
gradient_norm = np.linalg.norm(final_gradient)

print("Final gradient:", final_gradient)
print("Norm of final gradient:", gradient_norm)

num_of_iterations = res.nit
print("Number of iterations:", num_of_iterations)

# final point ---> x : [1, 1]
# final f value --->        fun: 2.0603335166685506e-11
# number of iterations ---> nit: 34