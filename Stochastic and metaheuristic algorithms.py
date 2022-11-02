import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize, differential_evolution
from pyswarm import pso


def fun(x):
    return 1 / (x ** 2 - 3 * x + 2)

def fun_ls(a, x, y):
    return (a[0] * x + a[1]) / (x ** 2 + a[2] * x + a[3]) - y

def aprox_fun(a, x, y):
    for i in range(0, len(x)):
        D = ((a[0] * x[i] + a[1]) / (x[i] ** 2 + a[2] * x[i] + a[3]) - y[i]) ** 2
    return D

# Creating a Function.
def normal_dist(x):
    prob_density = (np.pi * np.std(x)) * np.exp(-0.5 * ((x - np.mean(x)) / np.std(x)) ** 2)
    return prob_density

t = np.linspace(0, 1, 1001)
sigma = normal_dist(t)
x = list()
y = list()
for k in range(0, 1001):
    x.append(3 * k / 1000)
    if fun(x[k]) < (-100):
        y.append(sigma[k] - 100)
    elif -100 < fun(x[k]) < 100:
        y.append(fun(x[k]) + sigma[k])
    else:
        y.append(100 + sigma[k])

y = np.array(y)
x = np.array(x)
a1 = np.array([0, 1, 3, -0.15])
a2 = np.array([0, 1, 3, 2])
a3 = [(0.9, 1), (1.6, 1.7), (0, 0.2), (-6, -4.5)]
lb = [1, 1.3, 0.7, -6]
ub = [2, 1.4, 0.9, -5.8]
res_nm = minimize(aprox_fun, a1, args=(x, y), method='nelder-mead')
res_lm = least_squares(fun_ls, a2, args=(x, y), method='lm')
res_de = differential_evolution(aprox_fun, a3, args=(x, y))
res_PSO, fopt = pso(aprox_fun, lb, ub, args=(x, y))

print('Nelder-Mead: ', "a = %.2f, b = %.2f, c = %.2f, d = %.2f" % tuple(res_nm.x), )
print('Number of iterations: ', res_nm.nit)
print('Levenberg-Marquardt: ', "a = %.2f, b = %.2f, c = %.2f, d = %.2f" % tuple(res_lm.x))
print('Number of calculations of the function: ', res_lm.nfev)
print('Differential Evolution: ', "a = %.2f, b = %.2f, c = %.2f, d = %.2f" % tuple(res_de.x))
print('Number of iterations: ', res_de.nit)
print('Particle Swarm Optimization: ', "a = %.2f, b = %.2f, c = %.2f, d = %.2f" % tuple(res_PSO))
print('The number of iterations of the built-in method does not exceed 100.')

y1 = (res_nm.x[0]*x + res_nm.x[1])/(x**2 + res_nm.x[2]*x + res_nm.x[3])
y2 = (res_lm.x[0]*x + res_lm.x[1])/(x**2 + res_lm.x[2]*x + res_lm.x[3])
y3 = (res_de.x[0]*x + res_de.x[1])/(x**2 + res_de.x[2]*x + res_de.x[3])
y4 = (res_PSO[0]*x + res_PSO[1])/(x**2 + res_PSO[2]*x + res_PSO[3])

fig, ax = plt.subplots()
plt.plot(x, y, '.r', label='Generated data')
plt.plot(x, y1, '-b', label='Nelder-Mead')
plt.plot(x, y2, '-g', label='Levenberg-Marquardt')
plt.plot(x, y3, color = 'black', label='Differential Evolution')
plt.plot(x, y4, color = 'pink', label='Particle Swarm Optimization')
ax.legend()
ax.grid()
plt.ylim((-150, 150))
plt.show()
