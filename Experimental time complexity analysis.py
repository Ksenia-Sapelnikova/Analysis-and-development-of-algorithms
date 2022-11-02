import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from scipy.optimize import least_squares

def QuickSort(A):
    if len(A) <= 1:
        return A
    else:
        q = random.choice(A)
        L = []
        M = []
        R = []
        for elem in A:
            if elem < q:
                L.append(elem)
            elif elem > q:
                R.append(elem)
            else:
                M.append(elem)
        return QuickSort(L) + M + QuickSort(R)

def lin(a, x, y):
    return a[0] + a[1] * x - y

def sq(a, x, y):
    return a[0] + a[1] * x + a[2] * x**2 - y

def cubic(a, x, y):
    return a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3 - y

len_ar = list()
time1 = list()
practical_time = list()
fig, ax = plt.subplots()

#constant function
theoretical_time = list()
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        t1 = time.time_ns()
        y = 23*math.log10(137)+84*math.sin(32*math.pi)
        t2 = time.time_ns()
        time1.append(t2-t1)
    t = sum(time1)/len(time1)
    practical_time.append(t)
    theoretical_time.append(0)
plt.plot(len_ar, theoretical_time, '-g', label ='Analytical function y=0')
plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.ylabel('Algorithm operation time ns', fontsize=15)
plt.xlabel('Array Size', fontsize=15)
plt.title('Constant function', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
theoretical_time.clear()


#the sum of elements
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        local_sum = 0
        t1 = time.time_ns()
        for i in mass:
            local_sum = local_sum + i
        t2 = time.time_ns()
        time1.append(t2-t1)
    t = sum(time1)/len(time1)
    practical_time.append(t)
    mass = []

x = np.array(len_ar)
y = np.array(practical_time)
a0 = np.array([1, 1])

res_lsq = least_squares(lin, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label = 'Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('The sum of elements', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#the product of elements
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        prod=1
        t1 = time.time_ns()
        for i in mass:
            prod=prod*i
        t2 = time.time_ns()
        time1.append(t2-t1)
    t = sum(time1)/len(time1)
    practical_time.append(t)
    theoretical_time.append(massive)
    mass = []

x = np.array(len_ar)
y = np.array(practical_time)
a0 = np.array([1, 1])

res_lsq = least_squares(lin, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label = 'Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('The product of elements', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#direct calculation
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    x=0.9
    for it in (1, 5):
        if len(mass) == 1:
            t1 = time.time_ns()
            for i in range(len(mass)):
                ls = mass[i]*(x**i)
            t2 = time.time_ns()
            time1.append(float(t2-t1))
        else:
            t1 = time.time_ns()
            l = mass[0]+mass[1]*x
            xPower = x
            for i in range(2, len(mass)):
                xPower = xPower*x
                ls = ls + mass[i]*xPower
            t2 = time.time_ns()
            time1.append(float(t2-t1))
    t = sum(time1)/len(time1)
    practical_time.append(t)
    mass = []

x = np.array(len_ar)
y = np.array(practical_time)
a0 = np.array([1, 1])

res_lsq = least_squares(lin, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label = 'Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('Direct calculation', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#Horner’s method
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    x=0.9
    for it in (1, 5):
        ls=0
        t1 = time.time_ns()
        ls = mass[0] + x
        for i in range(1, len(mass)):
            ls = x*ls+mass[i]
        t2 = time.time_ns()
        time1.append(float(t2-t1))
    t = sum(time1)/len(time1)
    practical_time.append(t)
    theoretical_time.append(2*massive)
    mass = []

x = np.array(len_ar)
y = np.array(practical_time)
a0 = np.array([1, 1])

res_lsq = least_squares(lin, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label = 'Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('Horner’s method', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
theoretical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#Bubble Sort
for massive in range(1, 2001, 5):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        m1 = mass
        t1 = time.time_ns()
        for i in range(len(m1) - 1):
            for j in range(len(m1) - i - 1):
                if m1[j] > m1[j + 1]:
                    m1[j], m1[j + 1] = m1[j + 1], m1[j]
        t2 = time.time_ns()
        time1.append(t2-t1)
        m1 = []
    t = sum(time1)/len(time1)
    practical_time.append(t)
    theoretical_time.append(massive**2)
    mass = []

x = np.array(len_ar)
y = np.array(practical_time)
a0 = np.array([1, 1, 1])

res_lsq = least_squares(sq, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f, a2 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x, x ** 2])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label ='Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=15)
plt.xlabel('Array Size', fontsize=15)
plt.title('Bubble Sort', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
theoretical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#Quick Sort
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        m1 = mass
        t1 = time.time_ns()
        QuickSort(m1)
        t2 = time.time_ns()
        time1.append(float(t2-t1))
        m1 = []
    t = sum(time1)/len(time1)
    practical_time.append(t)
    theoretical_time.append(1.5*massive*math.log(massive))
    mass = []
y = np.log(len_ar)*len_ar*235.5

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(len_ar, y, '-g', label = 'Analytical function t=235.5*x*log(x)')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('Quick Sort', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
theoretical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#Timsort
for massive in range(1, 2001):
    len_ar.append(massive)
    mass = np.random.sample(massive)
    t1 = 0
    t2 = 0
    for it in (1, 5):
        m1 = mass
        t1 = time.time_ns()
        m1.sort()
        t2 = time.time_ns()
        time1.append(float(t2-t1))
        m1 = []
    t = sum(time1)/len(time1)
    practical_time.append(t)

y = np.log(len_ar)*len_ar*3.376

plt.plot(len_ar, practical_time, '-r', label = 'Practical results')
plt.plot(len_ar, y, '-g', label = 'Analytical function t=3.376*x*log(x)')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Array Size', fontsize=15)
plt.title('Timsort', fontsize=17)
ax.legend()
plt.show()
time1.clear()
len_ar.clear()
practical_time.clear()
x = []
y = []
x_p = []
y_p = []
a0 = []

#Matrix multiplication
t_0 = list()
t_for_plot = list()
size_mat = list()
for n in range(1, 501):
    size_mat.append(n)
    a = np.random.randint(100, size=(n, n))
    b = np.random.randint(100, size=(n, n))
    result_matrix = [[0 for i in range(n)] for i in range(n)]
    t1 = time.time_ns()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result_matrix[i][j] += a[i][k] * b[k][j]
    t2 = time.time_ns()
    t_for_plot.append((t2-t1))
x = np.array(size_mat)
y = np.array(t_for_plot)
a0 = np.array([1, 1, 1, 1])

res_lsq = least_squares(sq, x0=a0, args=(x, y))

print("a0 = %.2f, a1 = %.2f, a2 = %.2f, a2 = %.2f" % tuple(res_lsq.x))

f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x, x ** 2, x ** 3])])
x_p = np.linspace(min(x), max(x), 20)
y_p = f(x_p)

plt.plot(size_mat, t_for_plot, '-r', label = 'Practical results')
plt.plot(x_p, y_p, '-g', label ='Least squares method')
plt.ylabel('Algorithm operation time ns', fontsize=13)
plt.xlabel('Matrix Size', fontsize=15)
plt.title('Matrix multiplication', fontsize=17)
ax.legend()
plt.show()
x = []
y = []
x_p = []
y_p = []
a0 = []