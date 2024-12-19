import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Rosenbrock function parameters
a = 1
b = 100

def rosenbrock(x):
    # x is a numpy array with x[0] = x, x[1] = y
    X, Y = x[0], x[1]
    return (a - X)**2 + b*(Y - X**2)**2

def grad_rosenbrock(x):
    X, Y = x[0], x[1]
    # df/dx = 2(x - 1) - 400x(y - x²)
    # df/dy = 200(y - x²)
    dfdx = 2*(X - a) - 400*X*(Y - X**2)
    dfdy = 200*(Y - X**2)
    return np.array([dfdx, dfdy])

# Create a grid for plotting
x_vals = np.linspace(-2, 2, 1000)
y_vals = np.linspace(-1, 3, 1000)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z = rosenbrock([X_grid, Y_grid])

# Starting point
x0 = np.array([-1.0, 1.5])

# Stopping tolerance
tol = 1e-4
max_iter = 10000
alpha = 0.2  # Armijo parameter

ps = []  # to store visited points
ps.append(x0.copy())

xk = x0.copy()
gk = grad_rosenbrock(xk)
norm_gk = np.linalg.norm(gk)
iterations = 0

while norm_gk > tol and iterations < max_iter:
    iterations += 1
    
    # Start with t=1
    t = 1.0
    fk = rosenbrock(xk)
    gk = grad_rosenbrock(xk)
    norm_gk_sq = norm_gk**2

    w = xk - t*gk
    fw = rosenbrock(w)

    # Armijo line search
    if fw <= fk - alpha * t * norm_gk_sq:
        # Condition satisfied; try to increase t by doubling if possible
        while True:
            t_test = 2*t
            w_test = xk - t_test*gk
            fw_test = rosenbrock(w_test)
            if fw_test <= fk - alpha * t_test * norm_gk_sq:
                t = t_test
            else:
                break
    else:
        # Condition not satisfied; try halving t
        while fw > fk - alpha * t * norm_gk_sq:
            t /= 2
            w = xk - t*gk
            fw = rosenbrock(w)

    # Update xk
    xk = xk - t*gk
    ps.append(xk.copy())

    # Recompute gradient and norm
    gk = grad_rosenbrock(xk)
    norm_gk = np.linalg.norm(gk)

final_x = xk
final_f = rosenbrock(final_x)

# Compute the trajectory length
trajectory_length = 0.0
for i in range(len(ps)-1):
    step = ps[i+1] - ps[i]
    trajectory_length += np.linalg.norm(step)

print("Number of iterations:", iterations)        # should be 
print("Final point:", final_x)                  # should be close to [1, 1]
print("Final f value:", final_f)        # should be close to 0
print("Length of trajectory:", trajectory_length)

# Plot the results
plt.figure(1)
plt.contourf(X_grid, Y_grid, Z, 50, cmap=cm.jet)
plt.title('Rosenbrock')
plt.colorbar()

plt.figure(2)
plt.contourf(X_grid, Y_grid, np.log(Z), 50, cmap=cm.jet)
plt.title('Rosenbrock (log)')
plt.colorbar()

# Overlay the trajectory on the log-plot
plt.figure(3)
plt.contourf(X_grid, Y_grid, np.log(Z), 50, cmap=cm.jet)
plt.title('Steepest Descent on Rosenbrock (log scale)')
plt.colorbar()

px = np.array([p[0] for p in ps])
py = np.array([p[1] for p in ps])

plt.plot(px, py, 'w-', linewidth=2)  # Plot the path
plt.plot(px[0], py[0], 'o', color='black', label="Start")   # Starting point
plt.plot(px[-1], py[-1], 'o', color='white', label="End")   # End point
plt.xlim(-2, 2)
plt.ylim(-1, 3)
plt.legend()

plt.show()
