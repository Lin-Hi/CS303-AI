import random

import numpy as np
import argparse
import heapq

free = set()
routes = []

time_limit = -1
seed = -1
vertices_num = -1
depot = -1
origin_graph = [[]]
capacity = -1
min_dist = [[]]
edges_num = -1

random.seed(0)


def initialize():
    """
    从文件中读取所需信息。每个节点存 (cost, demand)
    """
    # 从命令行中读取数据
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str)
    parser.add_argument('-t', dest='time_limit', type=int)
    parser.add_argument('-s', dest='seed', type=int)
    args = parser.parse_args()
    global time_limit, seed
    file_path = args.file_name
    time_limit = args.time_limit
    seed = args.seed

    # 从文件中读取图信息
    global vertices_num, depot, capacity, edges_num, origin_graph, free
    f = open(file_path, 'r')
    next(f)
    vertices_num = int(f.readline().split(' : ')[1])
    depot = int(f.readline().split(' : ')[1])
    edges_num = int(f.readline().split(' : ')[1]) + int(f.readline().split(' : ')[1])
    next(f)
    capacity = int(f.readline().split(' : ')[1])
    next(f)
    next(f)
    origin_graph = [[() for _ in range(vertices_num + 1)] for _ in range(vertices_num + 1)]
    for _ in range(edges_num):
        start, end, cost, demand = list(map(int, f.readline().split()))
        origin_graph[start][end] = (cost, demand)
        origin_graph[end][start] = (cost, demand)
        if demand != 0:
            free.add((start, end, cost, demand))
            free.add((end, start, cost, demand))
    f.close()

    # 算两两端点之间最短路径
    global min_dist
    min_dist = [dijk(i) for i in range(0, vertices_num + 1)]
    return


def dijk(i: int):
    """
    求给定点到其他所有点的最短路径

    :param i: 给定点的编号
    :return: 列表，list[i] 为给定点到编号 i 的点的最短距离。i == 0 则返回空数组
    """
    if i == 0:
        return []
    ans = [float('inf') for _ in range(vertices_num + 1)]
    ans[i] = 0
    discovered = set()
    heap = []
    heapq.heappush(heap, (0, i))
    while len(discovered) != vertices_num:
        cur_dist, cur_i = heapq.heappop(heap)
        while cur_i in discovered:
            cur_dist, cur_i = heapq.heappop(heap)
        for neighbor_i in range(1, vertices_num + 1):
            if neighbor_i in discovered:
                continue
            if not origin_graph[cur_i][neighbor_i]:
                continue
            cost, demand = origin_graph[cur_i][neighbor_i]
            new_dist = cur_dist + cost
            old_dist = ans[neighbor_i]
            if new_dist < old_dist:
                ans[neighbor_i] = new_dist
                heapq.heappush(heap, (new_dist, neighbor_i))
        discovered.add(cur_i)
    return ans


def construct():
    """
    构造符合要求的最短路

    :return: 列表A，列表A的每个元素为列表B，每个列表B的元素固定两个：[0]route, [1]cost。
    route 为求出的路径的列表，列表中每个元素为一条单独的“出库-入库”路径，不包含首尾 depot。
    cost 为所有路径的总开销，包含首尾 depot
    """
    global free, min_dist, routes
    while len(free) != 0:
        route = []
        load = 0
        cost_route = 0
        i = depot
        dist_ = 0
        demand_ = 0
        cost_ = 0
        while not (len(free) == 0 or dist_ == float('inf')):
            dist_ = float('inf')
            edge_ = ()
            end_ = 0
            for start, end, cost, demand in free:
                if load + demand <= capacity:
                    edge = (start, end, cost, demand)
                    dist_i_start = min_dist[i][start]
                    if dist_i_start < dist_:
                        dist_ = dist_i_start
                        edge_ = edge
                        demand_ = demand
                        end_ = end
                        cost_ = cost
                    elif dist_i_start == dist_ and better(edge, edge_, load):
                        edge_ = edge
                        demand_ = demand
                        end_ = end
                        cost_ = cost
            if dist_ != float('inf'):
                route.append(edge_)
                free.remove(edge_)
                free.remove((edge_[1], edge_[0], edge_[2], edge_[3]))
                load += demand_
                cost_route += dist_ + cost_
                i = end_
        cost_route += min_dist[i][depot]
        routes.append([route, cost_route])
    return


def better(new_edge, old_edge, load) -> bool:
    """
    判断新路径是否比旧路径好

    :param new_edge: 新的路径
    :param old_edge: 新的路径
    :param load: 当前车辆负载量
    :return: boolean值，true则说明新路径更好
    """
    new_start, new_end, new_cost, new_demand = new_edge
    old_start, old_end, old_cost, old_demand = old_edge
    new_from_depot = min_dist[new_start][depot]
    old_from_depot = min_dist[old_start][depot]
    if load < capacity / 2:
        if new_from_depot > old_from_depot:
            return True
        else:
            return False
    else:
        if new_from_depot > old_from_depot:
            return False
        else:
            return True
    # return False


def print_out():
    """
    按照要求格式输出到控制台
    """
    routes_ans = 's '
    total_cost = 0
    for route, cost in routes:
        routes_ans += '0,'
        for start, end, _, _ in route:
            routes_ans += f'({start},{end}),'
        routes_ans += '0,'
        total_cost += cost
    routes_ans = routes_ans.strip(',')
    cost_ans = f'q {total_cost}'
    print(routes_ans)
    print(cost_ans)
    return


if __name__ == '__main__':
    initialize()
    construct()
    print_out()
