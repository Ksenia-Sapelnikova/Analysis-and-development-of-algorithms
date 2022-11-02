import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def sum_dist(iter, distance):
    sum = 0
    for i in range(len(iter) - 1):
        sum += distance[iter[i]][iter[i + 1]]

    sum += distance[iter[-1]][iter[0]]

    return sum


def simulate(coordinates, max_temp):
    first_t = []
    first_dist = 0
    first_iter = True

    n = len(coordinates)
    iter = random.sample(range(n), n)

    for temp in np.linspace(0.0001, 0.0005, max_temp)[::-1]:
        curr_dist = sum_dist(iter, distance)

        i, j = sorted(random.sample(range(n), 2))
        new_iter = iter.copy()
        new_iter[i], new_iter[j] = new_iter[j], new_iter[i]
        new_dist = sum_dist(new_iter, distance)

        if first_iter:
            first_t = new_iter.copy()
            first_dist = new_dist
            first_iter = False

        if (np.exp(curr_dist - new_dist) / temp) > random.random():
            iter = new_iter.copy()

    return first_t, first_dist, iter, sum_dist(iter, distance)


file_dist = 'lau15_dist.txt'
file_coord = 'lau15_xy.txt'

distance = pd.read_csv('lau15_dist.txt', sep=';').to_numpy()
print(distance)

coordinates = pd.read_csv('lau15_xy.txt', sep=';').to_numpy()
print(coordinates)

n = len(coordinates)

plt.scatter(coordinates[:, 0], coordinates[:, 1])
plt.show()

first_tour, first_dist, final_tour, final_dist = simulate(coordinates, 10000)

print('First iteration: ', first_tour)
print('Initial distance: ', first_dist)
print('Last iteration: ', final_tour)
print('Final distance: ', final_dist)

first_iter1 = list()
first_iter2 = list()
for i in first_tour:
    first_iter1.append(coordinates[i][0])
    first_iter2.append(coordinates[i][1])


plt.title('First iteration')
plt.scatter(coordinates[:, 0], coordinates[:, 1], s=30)
plt.plot(first_iter1, first_iter2)
plt.show()

final_iter1 = list()
final_iter2 = list()
for i in final_tour:
    final_iter1.append(coordinates[i][0])
    final_iter2.append(coordinates[i][1])

plt.title('Last iteration')
plt.scatter(coordinates[:, 0], coordinates[:, 1], s=30)
plt.plot(final_iter1, final_iter2)
plt.show()