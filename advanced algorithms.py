import math
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares

def sq(a, x, y):
    return a[0] + a[1] * x + a[2] * x**2 - y

def cubic(a, x, y):
    return a[0] + a[1] * x + a[2] * x**2 + a[3] * x**3 - y

def gen_adjlist(matrix):
    for s, nbrs in matrix.adjacency():
        line = str(s) + ": "
        list_of_edges = []
        for t, data in nbrs.items():
            line += str(t) + " "
            list_of_edges.append(t)
        yield list_of_edges

def show_graph(graph):
    plt.figure(num=None, figsize=(12, 8), dpi=80)
    plt.axis('off')
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.show()

def printMST(G,ln,nbr):
    print("Edge \tDistance")
    for i in range(1, ln):
        print (nbr[i], "-", i, "\t", G[i][ nbr[i] ] )

def prim(G):
    t1 = time.time()
    inf = math.inf
    ln = len(G)

    visited = [0] * ln
    nbr = [-1] * ln
    dist = [inf] * ln
    dist[0] = 0

    for _ in range(ln):

        minm = inf

        for u1 in range(ln):
            if dist[u1] < minm and not visited[u1]:
                minm = dist[u1]
                u = u1

        visited[u] += 1

        for v in range(ln):
            if((not visited[v]) and (dist[v] > G[u][v] > 0)):
                dist[v] = G[u][v]
                nbr[v] = u
    t2 = time.time()
    printMST(G, ln, nbr)
    return t2-t1

def FloydWarshall(G, n):
    t1 = time.time()
    W = [[[math.inf for j in range(n)] for i in range(n)] for k in range(n + 1)]

    for i in range(n):
        for j in range(n):
            W[0][i][j] = G[i][j]

    for k in range(1, n + 1):
        for i in range(n):
            for j in range(n):
                W[k][i][j] = min(W[k - 1][i][j], (W[k - 1][i][k - 1] + W[k - 1][k - 1][j]))
    t2 = time.time()
    return W[n], t2-t1




vertices = 40
edges = 160
matrix = np.zeros((vertices, vertices))
k = 0
while k != edges:
    i = random.randint(0, vertices - 1)
    j = random.randint(0, vertices - 1)
    if matrix[i][j] == 0 and i != j:
        weight = random.randint(1, 100)
        matrix[i][j] = weight
        matrix[j][i] = weight
        k += 1

print(prim(matrix))

sp, timeF = FloydWarshall(matrix, vertices)
print(timeF)
print("\nU -> V : W")
for i in range(vertices):
    for j in range(vertices):
        if (i != j):
            print(f"{i} -> {j} : {sp[i][j]}")

matrix = nx.from_numpy_matrix(matrix)

adjlist = []
for line in gen_adjlist(matrix):
    adjlist.append(line)

adjlist_dict = {}
for i in range(len(adjlist)):
    adjlist_dict[i] = adjlist[i]

show_graph(matrix)

# Estimation of time complexity
# len_ar = []
# time_prim = []
# time_Floyd = []
# for i in range(10, 200):
#     len_ar.append(i)
#     vertices = i
#     edges = i*4
#     matrix = np.zeros((vertices, vertices))
#     k = 0
#     while k != edges:
#         i = random.randint(0, vertices - 1)
#         j = random.randint(0, vertices - 1)
#         if matrix[i][j] == 0 and i != j:
#             weight = random.randint(1, 100)
#             matrix[i][j] = weight
#             matrix[j][i] = weight
#             k += 1
#
#     time_prim.append(prim(matrix))
#
#     sp, timeF = FloydWarshall(matrix, vertices)
#     time_Floyd.append(timeF)
#
# fig, ax = plt.subplots()
# x = np.array(len_ar)
# y = np.array(time_prim)
# a0 = np.array([100, 100, 100])
# res_lsq = least_squares(sq, x0=a0, args=(x, y))
# print("a0 = %.6f, a1 = %.6f, a2 = %.6f" % tuple(res_lsq.x))
# f = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x, x ** 2])])
# x_p = np.linspace(min(x), max(x), 20)
# y_p = f(x_p)
# plt.plot(len_ar, time_prim, '-r', label = 'Practical results')
# plt.plot(x_p, y_p, '-g', label = 'Least squares method')
# plt.ylabel('Algorithm operation time ns', fontsize=13)
# plt.xlabel('Array Size', fontsize=15)
# plt.title("Prim's Algorithm", fontsize=17)
# ax.legend()
# plt.show()

# x1 = np.array(len_ar)
# y1 = np.array(time_Floyd)
# a1 = np.array([1, 1, 1, 1])
# res_lsq = least_squares(cubic, x0=a1, args=(x1, y1))
# print("a0 = %.6f, a1 = %.6f, a2 = %.6f, a3 = %.6f" % tuple(res_lsq.x))
# f1 = lambda x: sum([u * v for u, v in zip(res_lsq.x, [1, x, x ** 2, x ** 3])])
# x_p1 = np.linspace(min(x), max(x), 20)
# y_p1 = f1(x_p1)
# plt.plot(len_ar, time_Floyd, '-r', label = 'Practical results')
# plt.plot(x_p1, y_p1, '-g', label = 'Least squares method')
# plt.ylabel('Algorithm operation time ns', fontsize=13)
# plt.xlabel('Array Size', fontsize=15)
# plt.title("Floyd-Warshall Algorithm", fontsize=17)
# ax.legend()
# plt.show()