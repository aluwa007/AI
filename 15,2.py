import numpy as np
import matplotlib.pyplot as plt

# Parameters
l = 1.0
T = 1.0
nx = 101  # Number of grid points in space
nt = 10000  # Number of grid points in time
dx = l / (nx - 1)
dt = T / (nt - 1)
alpha = 1.0
f = lambda x, t: np.sin(np.pi * x) * np.exp(-t)
p1 = 0.0
p2 = 0.0
m1 = 0.0
m2 = 0.0


x = np.linspace(0, l, nx)
u0 = np.sin(np.pi * x)


A = np.zeros((nx, nx))
A[0, 0] = 1.0
A[nx - 1, nx - 1] = 1.0
for i in range(1, nx - 1):
    A[i, i - 1] = -alpha * dt / dx ** 2
    A[i, i] = 1.0 + 2.0 * alpha * dt / dx ** 2
    A[i, i + 1] = -alpha * dt / dx ** 2

# Time-stepping
u = u0.copy()
for n in range(1, nt):
    b = u.copy()
    b[0] = m1 * dt / dx
    b[-1] = m2 * dt / dx
    b += dt * f(x, n * dt)
    u = np.linalg.solve(A, b)



def exact_solution(x, t):
    return np.sin(np.pi * x) * np.exp(-alpha * np.pi ** 2 * t)



t = T
u_exact = exact_solution(x, t)
plt.plot(x, u, 'b-', label='Numerical')
plt.plot(x, u_exact, 'r--', label='Exact')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()
