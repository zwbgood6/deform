'''sort the points along the rope'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt


def create_graph(points):
    '''create a graph based on 2 nearest neighbor
    points: Nx2 matrix
    T: graph
    '''    
    clf = NearestNeighbors(2).fit(points)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    return T

def find_max_length(paths):
    n = len(paths)
    max_length = 0
    for i in range(n):
        temp = len(paths[i])
        if temp > max_length:
            max_length = temp
    index = [] # maximum-lengh path's index
    for j in range(n):
        if max_length == len(paths[j]):
            index.append(j)
        else:
            index.append('None')
    return max_length, index    

def distance(P1, P2):
    '''compute L2 distance between points P1 and P2
    P1 = (x1, y1), P2 = (x2, y2)
    '''
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

def optimized_path(coords, start=None, start_idx=None):
    '''finds the nearest point to a point
    coords = [ [x1, y1], [x2, y2] , ...] 
    '''
    if start is None:
        start = coords[0]
        start_idx = 0
    coords = coords.tolist()
    start = start.tolist()
    pass_by = np.copy(coords)
    pass_by = pass_by.tolist()
    path = [start]
    path_idx = [start_idx]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: distance(path[-1], x))
        neareat_idx = coords.index(nearest)
        path.append(nearest)
        path_idx.append(neareat_idx)
        pass_by.remove(nearest)
    return path_idx, path

def find_start_node(T, points):
    '''find start node in a graph with minimum cost
    T: graph
    i: the (start/end) node 
    '''    
    paths = []
    coords = np.copy(points)
    for k in range(len(points)):
        index, _ = list(optimized_path(np.array(coords), start=np.array(coords[k]), start_idx=k))
        paths.append(index)
    #points = np.array(points)
    #paths = [list(optimized_path(points.tolist(), start=points[i].tolist())) for i in range(len(points))]
    #max_length, index = find_max_length(paths)# maximum length between paths
    min_dist = np.inf
    min_idx = 0
    for i in range(len(points)):
        #if index[i] == 'None':
        #    continue
        p = paths[i]
        ordered = points[p]
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < min_dist:
            min_dist = cost
            min_idx = i
    opt_order = paths[min_idx]        
    return min_idx, opt_order     

def main():
    # x = np.linspace(0, 2 * np.pi, 100)
    # y = np.sin(x)
    # idx = np.random.permutation(x.size)
    # x = x[idx]
    # y = y[idx]
    # points = np.c_[x, y]
    points = np.array([[3,6], [7,9], [4,6], [1,2], [5,6]])
    T = create_graph(points)
    i, opt_order = find_start_node(T, points)
    x, y = points[:,0], points[:,1]
    xx = x[opt_order]
    yy = y[opt_order]
    plt.plot(xx, yy)
    plt.show()


if __name__ == "__main__":
    main()