import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import time

# PART 1

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

def time_analize(node_start, algorithm):
    time_ex = []
    for i in range(10):
        start_time = time.time()
        algorithm
        path = algorithm(matrix, node_start)
        time_ex.append(time.time() - start_time)
    print('Average Time of work', round(sum(time_ex) / len(time_ex), 7))
    return path

def lenth_of_way(matrix, x):
    sum_way = 0
    if len(x)> 1:
        for i in range(1, len(x)):
            sum_way+= matrix[x[i-1]][x[i]]['weight']
    return sum_way

vertices = 100
edges = 500
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

matrix = nx.from_numpy_matrix(matrix)

adjlist = []
for line in gen_adjlist(matrix):
    adjlist.append(line)

adjlist_dict = {}
for i in range(len(adjlist)):
    adjlist_dict[i] = adjlist[i]

show_graph(matrix)

node_start = random.randint(0, vertices - 1)
print('Dijkstra`s algorithm')
print("Start vertex: ", node_start)
dijkstra = time_analize(node_start, nx.single_source_dijkstra_path)
for i in dijkstra:
    print("Final vertex: ", i, ': ', dijkstra[i], 'lenth of way: ', lenth_of_way(matrix, dijkstra[i]))
print('Bellman-Ford algorithm')
bellman_ford = time_analize(node_start, nx.single_source_bellman_ford_path)
for i in bellman_ford:
    print("Final vertex: ", i, ': ', bellman_ford[i], 'lenth of way: ', lenth_of_way(matrix, bellman_ford[i]))

# PART 2

def get_nodes():
    while True:
        i = random.randint(0, 9)
        j = random.randint(0, 19)
        start_node = (i, j)
        i = random.randint(0, 9)
        j = random.randint(0, 19)
        end_node = (i, j)
        if start_node in Graf and end_node in Graf:
            print(start_node, end_node)
            break
    return start_node, end_node

def draw_astar_path(path_astar):
    colors = list(mcolors.TABLEAU_COLORS)
    color_map = []
    for node in Graf:
        if node in path_astar:
            color_map.append(colors[1])
        else:
            color_map.append(colors[0])
    plt.figure(figsize=(12, 6))
    pos = {(x, y): (y, -x) for x, y in Graf.nodes()}
    nx.draw(Graf, pos=pos,
            node_color=color_map,
            with_labels=True,
            node_size=1400)
    plt.show()

data = np.ones((10, 20), dtype=int)
list_obstacles = []
while data.sum() != 160:
    i = random.randint(0, 9)
    j = random.randint(0, 19)
    if data[i][j] != 0:
        list_obstacles.append((i, j))
        data[i][j] = 0

print(data)

Graf = nx.grid_2d_graph(10, 20)
Graf.remove_nodes_from(list_obstacles)

for i in range(5):
    start_node, end_node = get_nodes()
    t1 = time.time()
    path_astar = nx.astar_path(Graf, start_node, end_node)
    t2 = time.time()
    print('Time of work', t2-t1)
    draw_astar_path(path_astar)