from heapq import heappop as hpop, heappush as hpush


def floyd(graph, start, end):
    """
    Args:
        graph: a given graph, as an adjacency matrix, the non-connected edge are set to infinity
        start: the index of the start vertex
        end: the index of the end vertex

    Returns:
        distance: the distance from the start point to the end point.

    sample:
    input (file):
    1          =>neighbors of node 0
    6          =>distances of (0,1)
    2 4        =>neighbors of node 1
    3 10       =>distances of (1,2) (1,4)
    3          =>neighbors of node 2
    7          =>distances of node (2,3)
    4          =>neighbors of node 3
    9          =>distances of node (3,4)
    0 3        =>neighbors of node 4
    3 9        =>distances of node (4,0) (4,3)

    graph: a distance matrix from above file
    [[0, 6.0, inf, inf, inf],
    [inf, 0, 3.0, inf, 10.0],
    [inf, inf, 0, 7.0, inf],
    [inf, inf, inf, 0, 9.0],
    [3.0, inf, inf, 9.0, 0]]

    output:
    16
    """

    ##############
    #  Please write your own code in the given space.
    #############

    # graph = [[]] # you should build a distance matrix (from the original graph) which include all the distances between any two vertexs.

    #############

    class Node:
        def __init__(self):
            self.neighbors = []
            self.weights = []
            self.has_out = False
            self.dist = float('inf')

    if start == end:
        return 0

    all_nodes = []
    for _ in range(len(graph)):
        all_nodes.append(Node())
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i == j:
                continue
            if graph[i][j] == float('inf'):
                continue
            all_nodes[i].neighbors.append(j)
            all_nodes[i].weights.append(int(graph[i][j]))

    heap = []
    all_nodes[start].dist = 0
    hpush(heap, (0, start))
    while len(heap) != 0:
        cur_idx = hpop(heap)[1]
        cur_node = all_nodes[cur_idx]
        if cur_node.has_out:
            continue
        cur_node.has_out = True
        for nei_idx in cur_node.neighbors:
            nei_node = all_nodes[nei_idx]
            if nei_node.has_out:
                continue
            wei = graph[cur_idx][nei_idx]
            if cur_node.dist + wei < nei_node.dist:
                nei_node.dist = cur_node.dist + wei
                hpush(heap, (nei_node.dist, nei_idx))

    return all_nodes[end].dist


if __name__ == '__main__':
    # test block
    for test_case in range(1,4):
        with open(f'test_cases/problem2/{test_case}.txt', 'r') as f:
            content = f.read().strip()
            lines = content.split('\n')
        n = len(lines) // 2
        origin_graph = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            origin_graph[i][i] = 0
        for i in range(n):
            neighbors = [*map(int, lines[2 * i].strip().split(' '))]
            distances = [*map(float, lines[2 * i + 1].strip().split(' '))]
            for j in range(len(neighbors)):
                k = neighbors[j]
                origin_graph[i][k] = distances[j]
        # print(graph)
        print(floyd(origin_graph, 0, n - 1))
