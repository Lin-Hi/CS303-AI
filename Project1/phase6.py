import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.15
FIRST_MAX_DEPTH = 3
SECOND_MAX_DEPTH = 5


# don't change the class name
# 修改原本错误的ab剪枝
# 根据剩余时间选择深度3还是深度5
# 修改原本错误的ab剪枝：删除传入的 color 参数
# 优化评估函数
# 修改 ab 所有 BUG
# ab 不再 copy 棋盘，自始至终都是一个棋盘
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.start_time = 0.0
        self.end_depth_5 = True
        self.time_out = time_out
        self.candidate_list = []
        self.max_depth = 3
        self.end_depth_5 = True
        self.end_depth_3 = False
        self.weight_1_2 = np.array([[24, -59, 41, -1, -1, 41, -59, 24], [-59, 7, 46, 53, 53, 46, 7, -59], [41, 46, 7, -20, -20, 7, 46, 41], [-1, 53, -20, -58, -58, -20, 53, -1], [-1, 53, -20, -58, -58, -20, 53, -1], [41, 46, 7, -20, -20, 7, 46, 41], [-59, 7, 46, 53, 53, 46, 7, -59], [24, -59, 41, -1, -1, 41, -59, 24]])
        self.weight_3 = np.array([[29, -42, -12, -11, -11, -12, -42, 29], [-42, -26, -35, 47, 47, -35, -26, -42], [-12, -35, -9, -53, -53, -9, -35, -12], [-11, 47, -53, -32, -32, -53, 47, -11], [-11, 47, -53, -32, -32, -53, 47, -11], [-12, -35, -9, -53, -53, -9, -35, -12], [-42, -26, -35, 47, 47, -35, -26, -42], [29, -42, -12, -11, -11, -12, -42, 29]])
        self.mobility_1_2 = np.array(
            [[1, 16, 19, 55, 55, 19, 16, 1], [16, 61, 26, -13, -13, 26, 61, 16], [19, 26, 46, 59, 59, 46, 26, 19],
             [55, -13, 59, 62, 62, 59, -13, 55], [55, -13, 59, 62, 62, 59, -13, 55], [19, 26, 46, 59, 59, 46, 26, 19],
             [16, 61, 26, -13, -13, 26, 61, 16], [1, 16, 19, 55, 55, 19, 16, 1]])
        self.mobility_3 = np.array(
            [[48, 10, 9, -26, -26, 9, 10, 48], [10, 55, 24, 1, 1, 24, 55, 10], [9, 24, -44, 14, 14, -44, 24, 9],
             [-26, 1, 14, 21, 21, 14, 1, -26], [-26, 1, 14, 21, 21, 14, 1, -26], [9, 24, -44, 14, 14, -44, 24, 9],
             [10, 55, 24, 1, 1, 24, 55, 10], [48, 10, 9, -26, -26, 9, 10, 48]])

    # The input is the current chessboard. Chessboard is a numpy array
    def go(self, chessboard):
        self.start_time = time.time()
        self.candidate_list.clear()
        self.candidate_list = self.get_valid_places(chessboard, self.color)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.max_depth = FIRST_MAX_DEPTH
        self.end_depth_5 = True
        self.end_depth_3 = False
        action_dep_3 = self.alpha_beta(chessboard, 0, self.color, float("-inf"), float("inf"))[1]
        self.max_depth = SECOND_MAX_DEPTH
        self.end_depth_3 = True
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

    def update_chessboard(self, chessboard, pos, color):
        def update_chessboard_line(chessboard, x, y, dx, dy, color):
            x_ = x + dx
            y_ = y + dy
            passed = []
            while 0 <= x_ < 8 and 0 <= y_ < 8:
                if chessboard[x_][y_] == -color:
                    passed.append((x_, y_))
                    x_ += dx
                    y_ += dy
                elif chessboard[x_][y_] == color:
                    return passed
                else:
                    return []
            return passed

        passed = []
        steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        x, y = pos
        for dx, dy in steps:
            passed += update_chessboard_line(chessboard, x, y, dx, dy, color)
        return passed

    def max_value(self, chessboard, depth, color, a, b):
        if self.end_depth_3 and time.time() - self.start_time > self.time_out - TIME_BEFORE_OUT:
            self.end_depth_5 = False
            return None, None
        if depth > self.max_depth:
            depth -= 1
            return self.calculate(chessboard), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard), None
        grades, action = float("-inf"), None
        oppo_color = -color
        overturn_chess = []
        for x, y in places:
            overturn_chess = self.update_chessboard(chessboard, (x, y), color)
            for i, j in overturn_chess:
                chessboard[i][j] = color
            chessboard[x][y] = color
            grades2, _ = self.min_value(chessboard, depth, oppo_color, a, b)
            if not self.end_depth_5:
                for i, j in overturn_chess:
                    chessboard[i][j] = oppo_color
                chessboard[x][y] = COLOR_NONE
                return None, None
            if grades2 > grades:
                action = x, y
                grades = grades2
            if grades >= b:
                for i, j in overturn_chess:
                    chessboard[i][j] = oppo_color
                chessboard[x][y] = COLOR_NONE
                # print('cut')
                break
            if grades > a:
                a = grades
            for i, j in overturn_chess:
                chessboard[i][j] = oppo_color
            chessboard[x][y] = COLOR_NONE
        return grades, action

    def min_value(self, chessboard, depth, color, a, b):
        if self.end_depth_3 and time.time() - self.start_time > self.time_out - TIME_BEFORE_OUT:
            self.end_depth_5 = False
            return None, None
        if depth > self.max_depth:
            depth -= 1
            return self.calculate(chessboard), None
        depth += 1
        places = self.get_valid_places(chessboard, color)
        if len(places) == 0:
            return self.calculate(chessboard), None
        grades, action = float("inf"), None
        oppo_color = -color
        overturn_chess = []
        for x, y in places:
            overturn_chess = self.update_chessboard(chessboard, (x, y), color)
            for i, j in overturn_chess:
                chessboard[i][j] = color
            chessboard[x][y] = color
            grades2, _ = self.max_value(chessboard, depth, oppo_color, a, b)
            if not self.end_depth_5:
                for i, j in overturn_chess:
                    chessboard[i][j] = oppo_color
                chessboard[x][y] = COLOR_NONE
                return None, None
            if grades2 < grades:
                action = x, y
                grades = grades2
            if grades <= a:
                for i, j in overturn_chess:
                    chessboard[i][j] = oppo_color
                chessboard[x][y] = COLOR_NONE
                # print('cut')
                break
            if grades < b:
                b = grades
            for i, j in overturn_chess:
                chessboard[i][j] = oppo_color
            chessboard[x][y] = COLOR_NONE
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
                return 9999999999999
            elif my_num > oppo_num:
                return -9999999999999
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
        c2 = [None, 43, 235, 250]
        c3 = [None, -28, -48, -64]
        c4 = [None, -23, -20, -4]
        result = c1 * location_weight + \
                 c2[state] * stable + \
                 c3[state] * diff_num + \
                 c4[state] * front + \
                 mobility
        return result
