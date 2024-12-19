import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Rosenbrock parameters
a = 1
b = 100

def rosenbrock(x):
    X, Y = x[0], x[1]
    return (a - X)**2 + b*(Y - X**2)**2

def grad_rosenbrock(x):
    X, Y = x[0], x[1]
    dfdx = 2*(X - a) - 400*X*(Y - X**2)
    dfdy = 200*(Y - X**2)
    return np.array([dfdx, dfdy])

def hessian_rosenbrock(x):
    X, Y = x[0], x[1]
    # Second derivatives
    d2fdx2 = 2 - 400*(Y - X**2) + 800*X**2
    d2fdy2 = 200
    d2fdxdy = -400*X
    d2fdydx = -400*X  # symmetric
    H = np.array([[d2fdx2, d2fdxdy],
                  [d2fdydx, d2fdy2]])
    return H

# Create the grid for plotting
x_vals = np.linspace(-2, 2, 1000)
y_vals = np.linspace(-1, 3, 1000)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
Z = rosenbrock([X_grid, Y_grid])

# Starting point
x0 = np.array([-1.0, 1.5])

tol = 1e-4
max_iter = 10000
xk = x0.copy()
ps = [xk.copy()]

gk = grad_rosenbrock(xk)
norm_gk = np.linalg.norm(gk)
iterations = 0

while norm_gk > tol and iterations < max_iter:
    iterations += 1
    H = hessian_rosenbrock(xk)
    
    # Solve H * p = g  => p = H^-1 g
    # Newton step: x_{k+1} = x_k - H^-1*g
    p = np.linalg.solve(H, gk)  # Solves H p = gk
    xk = xk - p
    
    ps.append(xk.copy())
    
    gk = grad_rosenbrock(xk)
    norm_gk = np.linalg.norm(gk)

final_x = xk
final_f = rosenbrock(final_x)

# Compute trajectory length
trajectory_length = 0.0
for i in range(len(ps)-1):
    step = ps[i+1] - ps[i]
    trajectory_length += np.linalg.norm(step)

print("Newton's Method Results:")
print("Number of iterations:", iterations)
print("Final point:", final_x)
print("Final f value:", final_f)
print("Trajectory length:", trajectory_length)

# Overlay the trajectory on log-plot in a new figure
plt.figure(4)
plt.contourf(X_grid, Y_grid, np.log(Z), 50, cmap=cm.jet)
plt.title('Newton\'s Method on Rosenbrock (log scale)')
plt.colorbar()

px = np.array([p[0] for p in ps])
py = np.array([p[1] for p in ps])

plt.plot(px, py, 'w-', linewidth=2)  # the path
plt.plot(px[0], py[0], 'o', color='black', label="Start")  # starting point
plt.plot(px[-1], py[-1], 'o', color='white', label="End")  # end point
plt.xlim(-2, 2)
plt.ylim(-1, 3)
plt.legend()

plt.show()
