import numpy as np
import matplotlib . pyplot as plt
from matplotlib import cm


def rosenbrock(V):
    # V is expected to be [X, Y] where X and Y can be arrays or scalars.
    X, Y = V
    a = 1
    b = 100
    return (a - X)**2 + b*(Y - X**2)**2

x = np.linspace(-2, 2, 1000)
y = np.linspace(-1, 3, 1000)
[X, Y] = np.meshgrid (x, y)
Z = rosenbrock([X, Y]) # To be implemented
plt.figure(1)
plt.contourf(X, Y, Z, 50, cmap=cm.jet )
plt.title('Rosenbrock')
plt.colorbar()
plt.figure(2)
plt.contourf(X, Y, np.log(Z), 50, cmap=cm.jet)
plt.title('Rosenbrock ( log )')
plt.colorbar 

plt.show()

# TRAAAAAAAAAAAAAAAAAAAAAASH
"""
plt.figure(3)
plt.contourf(X, Y, np.log(Z), 50, cmap=cm.jet)
plt.title('Steepest Descent')
plt.colorbar()
ps = # List of points with steepest descent . To be implemented
px = np.array([p[0] for p in ps])
py = np.array([ p [1] for p in ps ])
plt.plot(px, py)
plt.plot(px[0], py[0], 'o', color='black') # Starting point
plt.plot(px[-1], py[-1], 'o', color='white') # End point
plt.xlim(-2, 2)
plt.ylim(-1, 3)
"""