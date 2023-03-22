import random
import numpy as np
import argparse
import heapq
import time
import copy

time_limit = -1
seed = -1
vertices_num = -1
depot = -1
origin_graph = [[]]
capacity = -1
min_dist = [[]]
edges_num = -1
free = set()
population = []

start_time = time.time()
random.seed(time.time())

flip_per_routes = 5
single_insertion_per_routes = 5
multiple_insertion_per_routes = 5
swap_per_routes = 5
opt_per_routes = 5
population_size = 180
initial_generate = [400, population_size // 6]
new_generation_size = 180
later_generate = [400, new_generation_size // 6]
population_content = [120, 60]


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
    global vertices_num, depot, capacity, edges_num, origin_graph, \
        flip_per_routes, single_insertion_per_routes, multiple_insertion_per_routes, swap_per_routes, opt_per_routes
    f = open(file_path, 'r')
    next(f)
    vertices_num = int(f.readline().split(' : ')[1])
    depot = int(f.readline().split(' : ')[1])
    edges_num = int(f.readline().split(' : ')[1]) + int(f.readline().split(' : ')[1])
    [flip_per_routes,
     single_insertion_per_routes,
     multiple_insertion_per_routes,
     swap_per_routes,
     opt_per_routes] = \
        [edges_num // 9 for _ in range(5)]
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


def path_scanning(i=5):
    """
    :param i:使用哪种判定标准

    :return: 列表A，列表A的每个元素为列表B，每个列表B的元素为一条单独的“出库-入库”路径，不包含首尾 depot。
    """

    routes = []
    free_copy = free.copy()
    while len(free_copy) != 0:
        route = []
        load = 0
        i = depot
        dist_ = 0
        demand_ = 0
        while not (len(free_copy) == 0 or dist_ == float('inf')):
            dist_ = float('inf')
            edge_ = []
            end_ = 0
            free_list = list(free_copy)
            random.shuffle(free_list)
            for start, end, cost, demand in free_list:
                if load + demand <= capacity:
                    edge = [start, end, cost, demand]
                    dist_i_start = min_dist[i][start]
                    if dist_i_start < dist_:
                        dist_ = dist_i_start
                        edge_ = edge
                        demand_ = demand
                        end_ = end
                    elif dist_i_start == dist_ and better(edge, edge_, load, i):
                        edge_ = edge
                        demand_ = demand
                        end_ = end
            if dist_ != float('inf'):
                route.append(edge_)
                free_copy.remove(tuple(edge_))
                free_copy.remove((edge_[1], edge_[0], edge_[2], edge_[3]))
                load += demand_
                i = end_
        routes.append(route)
    return routes


def better(new_edge, old_edge, load, i) -> bool:
    """
    判断新路径是否比旧路径好

    :param new_edge: 新的路径
    :param old_edge: 新的路径
    :param load: 当前车辆负载量
    :param i: 按照哪一种方式判断
    :return: boolean值，true则说明新路径更好
    """

    def judge_1() -> bool:
        return new_from_depot > old_from_depot

    def judge_2() -> bool:
        return not judge_1()

    def judge_3() -> bool:
        return new_demand / new_cost > old_demand / old_cost

    def judge_4() -> bool:
        return not judge_3()

    def judge_5() -> bool:
        return judge_1() if load < capacity / 2 else judge_2()

    def judge_6() -> bool:
        return random.random() < 0.5

    new_start, new_end, new_cost, new_demand = new_edge
    old_start, old_end, old_cost, old_demand = old_edge
    new_from_depot = min_dist[new_start][depot]
    old_from_depot = min_dist[old_start][depot]
    if i == 1:
        return judge_1()
    elif i == 2:
        return judge_2()
    elif i == 3:
        return judge_3()
    elif i == 4:
        return judge_4()
    elif i == 5:
        return judge_5()
    else:
        return judge_6()


def calculate_cost(routes):
    total_cost = 0
    for route in routes:
        last_end = depot
        for start, end, cost, _ in route:
            # for start, end in route:
            total_cost += min_dist[last_end][start] + cost
            # total_cost += min_dist[last_end][start] + origin_graph[start][end][0]
            last_end = end
        total_cost += min_dist[last_end][depot]
    return total_cost


def print_out(routes, cost):
    """
    按照要求格式输出到控制台

    :param cost:
    :param routes: 内部为有demand边的走的顺序
    """
    routes_ans = 's '
    for route in routes:
        routes_ans += '0,'
        for start, end, _, _ in route:
            routes_ans += f'({start},{end}),'
        routes_ans += '0,'
    routes_ans = routes_ans.strip(',')
    cost_ans = f'q {cost}'
    print(routes_ans)
    print(cost_ans)
    return


def generate_population(each_generate, each_select):
    new_population = []
    for i in range(1, 7):
        h = []
        for _ in range(each_generate):
            routes = path_scanning(i)
            heapq.heappush(h, (calculate_cost(routes), routes))
        for _ in range(each_select):
            new_population.append(heapq.heappop(h))
    return new_population


def select(candidates, num):
    candidates.sort()
    return candidates[:num]


def select_best(candidates):
    best_cost, best_routes = float('inf'), []
    for cost, routes in candidates:
        if cost < best_cost:
            best_cost = cost
            best_routes = routes
    return best_cost, best_routes


def reproduce(population):
    flip_candidates = []
    single_insertion_candidates = []
    multiple_insertion_candidates = []
    swap_candidates = []
    opt_candidates = []
    for i in range(len(population)):
        r = random.random()
        if 0 <= r < 0.2:
            flip_candidates.append(population[i])
        elif 0.2 <= r < 0.4:
            single_insertion_candidates.append(population[i])
        elif 0.4 <= r < 0.6:
            multiple_insertion_candidates.append(population[i])
        elif 0.6 <= r < 0.8:
            swap_candidates.append(population[i])
        else:
            opt_candidates.append(population[i])
    for i in range(len(flip_candidates)):
        flip(flip_candidates[i], flip_per_routes)
    for i in range(len(single_insertion_candidates)):
        single_insertion(single_insertion_candidates[i], single_insertion_per_routes)
    for i in range(len(multiple_insertion_candidates)):
        multiple_insertion(multiple_insertion_candidates[i], multiple_insertion_per_routes)
    for i in range(len(swap_candidates)):
        swap(swap_candidates[i], swap_per_routes)
    for i in range(len(opt_candidates)):
        opt(opt_candidates[i], opt_per_routes)
    production = []
    all_candidates = flip_candidates + single_insertion_candidates + multiple_insertion_candidates + swap_candidates + opt_candidates
    for cost_routes in all_candidates:
        cost, routes = cost_routes
        if is_legal(routes):
            production.append(cost_routes)
    return production


def flip(cost_routes, times):
    """
    每次在所有出库入库路线中，随机找一条。
    在这条出库入库路线的所有边中，随机找一条翻转

    :param cost_routes:包含很多出库入库，格式为一个元组(所有出入库的cost，[[出库-入库线路],[出库入库线路]...])
    :param times: 对所有出库入库的路中，总的翻转次数
    :return:
    """
    routes = cost_routes[1]
    routes_num = len(routes)
    for _ in range(times):
        routes_i = random.randint(0, routes_num - 1)
        route = routes[routes_i]
        edges_num_in_route = len(route)
        edge_i = random.randint(0, edges_num_in_route - 1)
        edge = route[edge_i]
        start, end, _, _ = edge
        edge[0] = end
        edge[1] = start
        # pre = depot if edge_i == 0 else route[edge_i - 1][1]
        # next = depot if edge_i == edges_num_in_route - 1 else route[edge_i + 1][0]
        # cost_routes[0] = cost_routes[0] - min_dist[pre][start] - min_dist[end][next] + min_dist[pre][end] + \
        #                  min_dist[start][next]
    cost = calculate_cost(routes)
    return cost, routes


def single_insertion(cost_routes, times):
    """
    在所有出库入库路线中，随机找一条路线A。
    在这条路线上所有边中，随机找一条边a，把这条边e从这条路线中删除。
    然后再随机找一条出库入库路线B，把这条边e随机插入到一个B的一个索引处

    :param cost_routes:
    :param times: 需要插入的次数
    :return:
    """
    routes = cost_routes[1]
    routes_num = len(routes)
    for _ in range(times):
        last_routes_i = random.randint(0, routes_num - 1)
        last_route = routes[last_routes_i]
        edges_num_in_route = len(last_route)
        if edges_num_in_route <= 2:
            continue
        edge_i = random.randint(0, edges_num_in_route - 1)
        edge = last_route[edge_i]
        routes_i = random.randint(0, routes_num - 1)
        route = routes[routes_i]
        edges_num_in_route = len(route)
        if edges_num_in_route <= 2 and routes_i != last_routes_i:
            continue
        last_route.pop(edge_i)
        edge_i = random.randint(0, edges_num_in_route - 1)
        route.insert(edge_i, edge)
    cost = calculate_cost(routes)
    return cost, routes


def multiple_insertion(cost_routes, times):
    """
    在所有出库入库路线中，随机找一条路线A。
    在这条路线上所有边中，随机找一段边[e1, e2, e3...]，把这段边e从这条路线中删除。
    然后再随机找一条出库入库路线B，把这段边[e1, e2, e3...]随机插入到一个B的一个索引处
    :param cost_rout]s:
    :param times:
    :return:
    """
    routes = cost_routes[1]
    routes_num = len(routes)
    for _ in range(times):
        last_routes_i = random.randint(0, routes_num - 1)
        last_route = routes[last_routes_i]
        edges_num_in_route = len(last_route)
        if edges_num_in_route <= 2:
            continue
        edge_i = random.randint(0, edges_num_in_route - 2)
        edge = last_route[edge_i:edge_i + 2]
        routes_i = random.randint(0, routes_num - 1)
        route = routes[routes_i]
        edges_num_in_route = len(route)
        if edges_num_in_route <= 2 and routes_i != last_routes_i:
            continue
        del last_route[edge_i:edge_i + 2]
        edge_i = random.randint(0, edges_num_in_route - 1)
        routes[routes_i] = route[:edge_i] + edge + route[edge_i:]
    cost = calculate_cost(routes)
    return cost, routes


def swap(cost_routes, times):
    routes = cost_routes[1]
    routes_num = len(routes)
    for _ in range(times):
        routes_i = random.randint(0, routes_num - 1)
        route = routes[routes_i]
        edges_num_in_route = len(route)
        edge_i = random.randint(0, edges_num_in_route - 1)
        edge_1 = route[edge_i]
        routes_i = random.randint(0, routes_num - 1)
        route = routes[routes_i]
        edges_num_in_route = len(route)
        edge_i = random.randint(0, edges_num_in_route - 1)
        edge_2 = route[edge_i]
        a, b, c, d = edge_1
        edge_1 = (edge_2[0], edge_2[1], edge_2[2], edge_2[3])
        edge_2 = (a, b, c, d)
    cost = calculate_cost(routes)
    return cost, routes


def opt(cost_routes, times):
    """

    :param cost_routes:
    :param times:
    :return:
    """
    routes = cost_routes[1]
    routes_num = len(routes)
    if routes_num < 2:
        return
    for _ in range(times):
        routes_i = random.sample(range(routes_num), 2)
        route = (routes[routes_i[0]], routes[routes_i[1]])
        edges_num_in_route = (len(route[0]), len(route[1]))
        if edges_num_in_route[0] < 2 or edges_num_in_route[1] < 2:
            continue
        edge_i = (random.randint(1, edges_num_in_route[0] - 1), random.randint(1, edges_num_in_route[1] - 1))
        sub_1_front, sub_1_end = route[0][:edge_i[0]], route[0][edge_i[0]:]
        sub_2_front, sub_2_end = route[1][:edge_i[1]], route[1][edge_i[1]:]
        new_route_11 = sub_1_front + sub_2_end
        new_route_12 = sub_2_front + sub_1_end
        sub_2_front.reverse()
        sub_1_end.reverse()
        new_route_21 = sub_1_front + sub_2_front
        new_route_22 = sub_1_end + sub_2_end
        cost_0 = calculate_cost(list(route))
        cost_1 = calculate_cost([new_route_11, new_route_12])
        cost_2 = calculate_cost([new_route_21, new_route_22])
        flag = 0
        min_cost = cost_0
        if cost_1 < min_cost:
            min_cost = cost_1
            flag = 1
        if cost_2 < min_cost:
            flag = 2
        if flag != 0:
            routes.pop(max(routes_i[0], routes_i[1]))
            routes.pop(min(routes_i[0], routes_i[1]))
            if flag == 1:
                routes.append(new_route_11)
                routes.append(new_route_12)
            else:
                routes.append(new_route_21)
                routes.append(new_route_22)
    cost = calculate_cost(routes)
    return cost, routes


def is_legal(routes):
    for route in routes:
        load = 0
        for edge in route:
            _, _, _, demand = edge
            # start,end = edge
            # demand = origin_graph[start][end][1]
            load += demand
            if load > capacity:
                return False
    return True


if __name__ == '__main__':
    initialize()
    # a=0
    # while True:
    #     population = generate_population(initial_generate[0], initial_generate[1])
    #     population = reproduce(population)
    #     best_cost, best_routes = select_best(population)
    #     a += 1
    #     print(a)
    population = generate_population(initial_generate[0], initial_generate[1])
    best_cost, best_routes = copy.deepcopy(select_best(population))
    i = 0
    while True:
        if time.time() - start_time > time_limit - 5:
            break
        # print("Before reproduce:")
        # print(best_routes)
        population = reproduce(population)
        # print("After reproduce:")
        # print(best_routes)
        new_generation = generate_population(later_generate[0], later_generate[1])
        cur_best = population[0] if population[0][0] < new_generation[0][0] else new_generation[0]
        if cur_best[0] < best_cost:
            best_cost = cur_best[0]
            best_routes = copy.deepcopy(cur_best[1])
        next_generation = population[:population_content[0]] + new_generation[:population_content[1]]
        population = next_generation
        i += 1
        # print(f'{i}th: {best_cost}')
    # print(is_legal(best_routes))
    print_out(best_routes, best_cost)
