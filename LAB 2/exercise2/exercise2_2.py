# Steepest descent method with Armijo line search

import numpy as np

# Define the function f and its gradient
def f(x):
    # x is a numpy array: x[0] = x, x[1] = y
    # f(x,y) = (x+1)^2 - x*y + 3(y-5)^2
    return (x[0] + 1)**2 - x[0]*x[1] + 3*(x[1] - 5)**2

def grad_f(x):
    # Gradient of f:
    # f_x = 2x - y + 2
    # f_y = -x + 6y - 30
    g = np.zeros_like(x)
    g[0] = 2*x[0] - x[1] + 2
    g[1] = -x[0] + 6*x[1] - 30
    return g

# The optimal point found analytically:
x_star = np.array([18/11, 58/11])
f_star = f(x_star)

# Steepest Descent with Armijo line search
xk = np.array([1.0, 1.0])  # starting point
tol = 1e-3
max_iter = 10000

alpha = 0.2  # Armijo parameter (given in instructions)
gk = grad_f(xk)
norm_gk = np.linalg.norm(gk)
iterations = 0

while norm_gk > tol and iterations < max_iter:
    iterations += 1
    
    # Start with t=1
    t = 1.0
    fk = f(xk)
    gk = grad_f(xk)
    norm_gk_sq = norm_gk**2
    
    # Candidate next point
    w = xk - t*gk
    fw = f(w)
    
    # Check Armijo condition
    # We have two paths: If f(w(t)) <= f(x_k) - 0.2*t*||g||^2, try to enlarge t
    # Otherwise, reduce t until condition is met.
    
    if fw <= fk - alpha * t * norm_gk_sq:
        # Condition is satisfied; try to increase t by doubling
        # until it no longer satisfies, then step back one doubling.
        while True:
            t_test = 2*t
            w_test = xk - t_test*gk
            fw_test = f(w_test)
            if fw_test <= fk - alpha * t_test * norm_gk_sq:
                t = t_test
            else:
                break
        # Use the largest t found
    else:
        # Condition not satisfied; try halving t until it is.
        while fw > fk - alpha * t * norm_gk_sq:
            t = t/2
            w = xk - t*gk
            fw = f(w)
    
    # Update xk
    xk = xk - t*gk
    
    # Recompute gradient and norm for next iteration
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

