# NEWTON'S METHOD FOR UNCONSTRAINED OPTIMIZATION 

# SHOUTOUT TO MY BRO NEWTON FOR HELPING ME OUT WITH THIS ONE

import numpy as np

# Define the function f and its gradient
def f(x):
    # x[0] = x, x[1] = y
    return (x[0] + 1)**2 - x[0]*x[1] + 3*(x[1] - 5)**2

def grad_f(x):
    g = np.zeros_like(x)
    g[0] = 2*x[0] - x[1] + 2   # df/dx
    g[1] = -x[0] + 6*x[1] - 30 # df/dy
    return g

# Optimal point known from analysis
x_star = np.array([18/11, 58/11])
f_star = f(x_star)

# Hessian (constant)
H = np.array([[2, -1],
              [-1, 6]])

# Inverse of H
det = 2*6 - (-1)*(-1) # =11
H_inv = (1/det)*np.array([[6, 1],
                          [1, 2]])

# Newton's method
xk = np.array([1.0, 1.0])  # starting point
tol = 1e-3
max_iter = 1000

iterations = 0
gk = grad_f(xk)
norm_gk = np.linalg.norm(gk)

while norm_gk > tol and iterations < max_iter:
    iterations += 1
    
    # Newton step
    # x_{k+1} = x_k - H^-1 * grad_f(x_k)
    xk = xk - H_inv.dot(gk)
    
    gk = grad_f(xk)
    norm_gk = np.linalg.norm(gk)

final_x = xk
final_f = f(final_x)
error = np.abs(final_f - f_star)

print("Number of iterations until ||grad_f(x)|| < 1e-3:", iterations)
print("Final point:", final_x)
print("Final f value:", final_f)
print("Optimal f value:", f_star)
print("Absolute error with respect to optimal f value:", error)
