import numpy as np
import numba as nb
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.15


# don't change the class name
# 修改原本错误的ab剪枝
# 根据剩余时间选择深度3还是深度5
# 修改原本错误的ab剪枝：删除传入的 color 参数
# 优化评估函数
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.start_time = 0.0
        self.end_depth_5 = True
        self.time_out = time_out
        self.candidate_list = []
        self.MAX_DEPTH = 3
        self.weight_1_2 = np.array([
            [-2, -45, 36, -12, -12, 36, -45, -2],
            [-45, 59, 61, 62, 62, 61, 59, -45],
            [36, 61, 35, -55, -55, 35, 61, 36],
            [-12, 62, -55, -61, -61, -55, 62, -12],
            [-12, 62, -55, -61, -61, -55, 62, -12],
            [36, 61, 35, -55, -55, 35, 61, 36],
            [-45, 59, 61, 62, 62, 61, 59, -45],
            [-2, -45, 36, -12, -12, 36, -45, -2]
        ])
        self.weight_3 = np.array([
            [-5, -38, -23, -59, -59, -23, -38, -5],
            [-38, 61, 13, 56, 56, 13, 61, -38],
            [-23, 13, -14, -30, -30, -14, 13, -23],
            [-59, 56, -30, -51, -51, -30, 56, -59],
            [-59, 56, -30, -51, -51, -30, 56, -59],
            [-23, 13, -14, -30, -30, -14, 13, -23],
            [-38, 61, 13, 56, 56, 13, 61, -38],
            [-5, -38, -23, -59, -59, -23, -38, -5]
        ])
        self.mobility_1_2 = np.array([
            [18, 13, 14, 44, 44, 14, 13, 18],
            [13, 60, 2, 48, 48, 2, 60, 13],
            [14, 2, 27, -36, -36, 27, 2, 14],
            [44, 48, -36, -17, -17, -36, 48, 44],
            [44, 48, -36, -17, -17, -36, 48, 44],
            [14, 2, 27, -36, -36, 27, 2, 14],
            [13, 60, 2, 48, 48, 2, 60, 13],
            [18, 13, 14, 44, 44, 14, 13, 18]
        ])
        self.mobility_3 = np.array([
            [37, 57, 51, 24, 24, 51, 57, 37],
            [57, -58, -54, 50, 50, -54, -58, 57],
            [51, -54, 47, -44, -44, 47, -54, 51],
            [24, 50, -44, -36, -36, -44, 50, 24],
            [24, 50, -44, -36, -36, -44, 50, 24],
            [51, -54, 47, -44, -44, 47, -54, 51],
            [57, -58, -54, 50, 50, -54, -58, 57],
            [37, 57, 51, 24, 24, 51, 57, 37]
        ])

    # The input is the current chessboard. Chessboard is a numpy array
    def go(self, chessboard):
        self.start_time = time.time()
        self.candidate_list.clear()
        self.candidate_list = self.get_valid_places(chessboard, self.color)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.MAX_DEPTH = 3
        self.end_depth_5 = True
        action_dep_3 = self.alpha_beta(chessboard, 0, self.color, float("-inf"), float("inf"))[1]
        self.MAX_DEPTH = 5
        action_dep_5 = self.alpha_beta(chessboard, 0, self.color, float("-inf"), float("inf"))[1]
        if not self.end_depth_5:
            self.candidate_list.append(action_dep_3)
            # print(3)
        else:
            self.candidate_list.append(action_dep_5)
            # print(5)
        return self.candidate_list

    def can_go(self, chessboard, color, x, y):
        # (x, y) is COLOR_NONE
        FINDING_OPPO = 0
        FINDING_SELF = 1
        oppo_color = -color
        choice = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in choice:
            tx, ty = x, y
            state = FINDING_OPPO
            while True:
                tx += dx
                ty += dy
                if not (0 <= tx < len(chessboard) and 0 <= ty < len(chessboard[0])):
                    break
                if state == FINDING_OPPO:
                    if chessboard[tx][ty] == oppo_color:
                        state = FINDING_SELF
                    else:
                        break
                else:
                    # state == FINDING_SELF
                    if chessboard[tx][ty] == color:
                        return True
                    if chessboard[tx][ty] == oppo_color:
                        continue
                    if chessboard[tx][ty] == COLOR_NONE:
                        break
        return False

    def get_valid_places(self, chessboard, color):
        idx = np.where(chessboard == COLOR_NONE)
        empty_places = list(zip(idx[0], idx[1]))
        valid_places = []
        for x, y in empty_places:
            if self.can_go(chessboard, color, x, y):
                valid_places.append((x, y))
        return valid_places

    def max_value(self, chessboard, depth, color, a, b):
        if time.time() - self.start_time > self.time_out - 0.2:
            self.end_depth_5 = False
            return self.calculate(chessboard), None
        if depth > self.MAX_DEPTH:
            depth -= 1
            return self.calculate(chessboard), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard), None
        grades, action = float("-inf"), None
        next_color = -color
        for x, y in places:
            next_chessboard = chessboard.copy()
            next_chessboard[x][y] = next_color
            grades2, _ = self.min_value(next_chessboard, depth, next_color, a, b)
            if grades2 > grades:
                action = x, y
                grades = grades2
            if grades >= b:
                break
            if grades > a:
                a = grades
        return grades, action

    def min_value(self, chessboard, depth, color, a, b):
        if time.time() - self.start_time > self.time_out - TIME_BEFORE_OUT:
            self.end_depth_5 = False
            return self.calculate(chessboard), None
        if depth > self.MAX_DEPTH:
            depth -= 1
            return self.calculate(chessboard), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard), None
        grades, action = float("inf"), None
        next_color = -color
        for x, y in places:
            next_chessboard = chessboard.copy()
            next_chessboard[x][y] = next_color
            grades2, _ = self.max_value(next_chessboard, depth, next_color, a, b)
            if grades2 < grades:
                action = x, y
                grades = grades2
            if grades <= a:
                break
            if grades < b:
                b = grades
        return grades, action

    def alpha_beta(self, chessboard, depth, color, a, b):
        return self.max_value(chessboard, depth, color, a, b)

    def get_stable_disc(self, chessboard, stable_board):
        my_stable = 0
        oppo_stable = 0
        if chessboard[0][0] != COLOR_NONE:
            is_my_chess = chessboard[0][0] == self.color
            max_width = 7
            line = 0
            while max_width >= 0 and 0 <= line <= 7:
                col = 0
                for col in range(max_width + 1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        stable_board[line][col] = True
                        my_stable += 1
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col -= 1
                        break
                max_width = col
                line += 1
        if chessboard[7][0] != COLOR_NONE:
            is_my_chess = chessboard[7][0] == self.color
            max_width = 7
            line = 7
            while max_width >= 0 and 0 <= line <= 7:
                col = 0
                for col in range(max_width + 1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col -= 1
                        break
                max_width = col
                line -= 1
        if chessboard[0][7] != COLOR_NONE:
            is_my_chess = chessboard[0][7] == self.color
            min_width = 0
            line = 0
            while min_width <= 7 and 0 <= line <= 7:
                col = 7
                for col in range(7, min_width - 1, -1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col += 1
                        break
                min_width = col
                line += 1
        if chessboard[7][7] != COLOR_NONE:
            is_my_chess = chessboard[7][7] == self.color
            min_width = 0
            line = 7
            while min_width <= 7 and 0 <= line <= 7:
                col = 7
                for col in range(7, min_width - 1, -1):
                    if is_my_chess and chessboard[line][col] == self.color:
                        my_stable += 1
                        stable_board[line][col] = True
                    elif not is_my_chess and chessboard[line][col] == -self.color:
                        oppo_stable += 1
                        stable_board[line][col] = True
                    else:
                        col += 1
                        break
                min_width = col
                line -= 1
        return (my_stable, oppo_stable)

    def calculate(self, chessboard):

        # 判断前中后状态
        my_num = 0
        oppo_num = 0
        for line in chessboard:
            for chess in line:
                if chess == self.color:
                    my_num += 1
                elif chess == -self.color:
                    oppo_num += 1
        total_num = my_num + oppo_num
        diff_num = my_num - oppo_num
        state = 0
        # 0：未赋值     1： 开局     2：中局     3：残局
        if 0 <= total_num < 25:
            state = 1
        elif 25 <= total_num < 45:
            state = 2
        else:
            state = 3

        # 判断终局
        valid_my = self.get_valid_places(chessboard, self.color)
        valid_oppo = self.get_valid_places(chessboard, -self.color)
        if len(valid_my) == 0 and len(valid_oppo) == 0:
            if my_num < oppo_num:
                return 9999999999
            elif my_num > oppo_num:
                return -9999999999
            else:
                return 0

        # 求权值和
        location_weight = 0
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if state != 3:
                    if chessboard[i][j] == self.color:
                        location_weight -= self.weight_1_2[i][j]
                    elif chessboard[i][j] == -self.color:
                        location_weight += self.weight_1_2[i][j]
                else:
                    if chessboard[i][j] == self.color:
                        location_weight -= self.weight_3[i][j]
                    elif chessboard[i][j] == -self.color:
                        location_weight += self.weight_3[i][j]
        # location_weight 计算完毕

        # 算行动力，用 8x8 数组表示行动力
        mobility = 0
        valid_my = self.get_valid_places(chessboard, self.color)
        valid_oppo = self.get_valid_places(chessboard, -self.color)
        for x, y in valid_my:
            if state != 3:
                mobility += self.mobility_1_2[x][y]
            else:
                mobility += self.mobility_3[x][y]
        for x, y in valid_oppo:
            if state != 3:
                mobility -= self.mobility_1_2[x][y]
            else:
                mobility -= self.mobility_3[x][y]
        # mobility 计算完毕

        # 计算双方稳定子和差值
        stable_board = [
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False]
        ]
        stable_my, stable_oppo = self.get_stable_disc(chessboard, stable_board)
        stable = stable_oppo - stable_my
        # stable 计算完毕

        # 计算边缘子
        frontier_my = 0
        frontier_oppo = 0
        direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for i in range(self.chessboard_size):
            for j in range(self.chessboard_size):
                if chessboard[i][j] != COLOR_NONE and not stable_board[i][j]:
                    for dx, dy in direction:
                        x = i + dx
                        y = j + dy
                        if 0 <= x <= 7 and 0 <= y <= 7 and chessboard[x][y] == COLOR_NONE:
                            if chessboard[i][j] == self.color:
                                frontier_my += 1
                            else:
                                frontier_oppo += 1
        front = frontier_my - frontier_oppo
        # front 计算完毕

        c1 = 1
        c2 = [None, 24, 123, 192]
        c3 = [None, -63, -51, -50]
        c4 = [None, -4, -40, -68]
        result = c1 * location_weight + \
                 c2[state] * stable + \
                 c3[state] * diff_num + \
                 c4[state] * front + \
                 mobility
        return result
