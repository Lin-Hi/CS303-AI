"""
Example:
input:
map:
---------
------x--
-x-------
---@-----
---##----
------x--
--x----x-
-x-------
---------
action:
0 0 3 3 0 3 3 1 1 1 1 1 3 1 1 2 2 2 2 2

output:
7 3
"""

from collections import deque

next_body = ()


def has_next_body(graph, cur_pos: (int, int)) -> bool:
    global next_body
    for i in range(4):
        if symbol(graph, next_pos(cur_pos, i)) == '#':
            next_body = next_pos(cur_pos, i)
            set_symbol(graph, next_body, '*')
            return True
    return False


def next_pos(cur_pos: (int, int), direction: int) -> (int, int):
    i, j = cur_pos
    if direction == 0:
        return i - 1, j
    elif direction == 1:
        return i + 1, j
    elif direction == 2:
        return i, j - 1
    else:
        return i, j + 1


def symbol(graph, pos: (int, int)) -> str:
    x, y = pos
    return graph[x - 1][y - 1]


def set_symbol(graph, pos, symbol):
    x, y = pos
    graph[x - 1][y - 1] = symbol


if __name__ == '__main__':
    # test block, you may need to modify this block.
    for test_case in range(1, 5):
        with open(f'test_cases/problem3/{test_case}-map.txt', 'r') as f:
            game_map = [list(line.strip()) for line in f.readlines()]
        # print(game_map)
        with open(f'./test_cases/problem3/{test_case}-actions.txt', 'r') as f:
            actions = [*map(int, f.read().split(' '))]
        # print(actions)

        barrier = set()

        for i in range(len(game_map)):
            for j in range(len(game_map[i])):
                if game_map[i][j] == 'x':
                    barrier.add((i + 1, j + 1))
        for i in range(len(game_map[0]) + 2):
            barrier.add((0, i))
            barrier.add((len(game_map) + 1, i))
        for i in range(len(game_map) + 2):
            barrier.add((i, 0))
            barrier.add((i, len(game_map[0]) + 1))

        cur_pos = ()
        for i in range(len(game_map)):
            for j in range(len(game_map[0])):
                if game_map[i][j] == '@':
                    cur_pos = (i + 1, j + 1)
                    break

        snake = deque()
        snake.append(cur_pos)
        head_pos = cur_pos
        while has_next_body(game_map, cur_pos):
            snake.append(next_body)
            cur_pos = next_body

        counter = 0
        has_succeed = True
        for direction in actions:
            counter += 1
            new_pos = next_pos(head_pos, direction)
            if new_pos in barrier or new_pos in snake:
                print(counter)
                has_succeed = Falsedanshizhende
                break
            snake.appendleft(new_pos)
            snake.pop()
            head_pos = new_pos
        if has_succeed:
            print(f'{head_pos[0] - 1} {head_pos[1] - 1}')
