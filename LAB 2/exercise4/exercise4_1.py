import numpy as np
from scipy.optimize import minimize

# Define the original function
def f(xy):
    x, y = xy
    return (2*x + y)**2 + y**4 - 4*x - 4*y

# Define the constraints g_i(x)
def g1(xy):
    x, y = xy
    return x**2 + y**2 - 4  # <= 0

def g2(xy):
    x, y = xy
    return 5 - (4*x + 5*y)  # <= 0

# Penalty function alpha(x)
def alpha(xy):
    val_g1 = max(0, g1(xy))
    val_g2 = max(0, g2(xy))
    return val_g1**2 + val_g2**2

# Combined penalized function F_mu(x)
def F(xy, mu):
    return f(xy) + mu * alpha(xy)

# Initial guess: A feasible point, for example (0,1)
x0 = np.array([0.0, 1.0])       # This was found by just looking at the constraints

mu = 0.1
beta = 10.0
tolerance = 1e-4

iterations = 0
while True:
    iterations += 1
    # Solve unconstrained subproblem with current mu
    res = minimize(F, x0, args=(mu,))
    
    xk = res.x
    val_mu_alpha = mu * alpha(xk)
    
    print(f"Iteration {iterations}: mu={mu}, xk={xk}, f(xk)={f(xk)}, mu*alpha(xk)={val_mu_alpha}")

    if val_mu_alpha < tolerance:
        # Stopping condition reached
        break
    else:
        # Increase mu and repeat
        mu *= beta
        # Use the current solution as the starting point for the next iteration
        x0 = xk

# After the loop
final_x = xk
final_f = f(final_x)
print("\nFinal Results:")
print("Optimal point:", final_x)
print("Approx. optimal function value:", final_f)
print("Number of outer iterations (penalty updates):", iterations)
