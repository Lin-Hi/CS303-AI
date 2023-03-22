import numpy as np
import numba as nb
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(time.time())
TIME_BEFORE_OUT = 0.3
FIRST_MAX_DEPTH = 1
SECOND_MAX_DEPTH = 3
THIRD_MAX_DEPTH = 5


# don't change the class name
# 修改原本错误的ab剪枝
# 根据剩余时间选择深度3还是深度5
# 修改原本错误的ab剪枝：删除传入的 color 参数
# 优化评估函数
# 修改 ab 所有 BUG
# 搜层顺序：1 -> 3 -> 5，此版本为交到 OJ 的版本
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.start_time = 0.0
        self.time_out = time_out
        self.candidate_list = []
        self.depth_1_state = 0
        self.depth_3_state = 0
        self.depth_4_state = 0
        self.max_depth = 1
        self.weight_1_2 = np.array(
            [[58, -52, 47, 39, 39, 47, -52, 58], [-52, 16, 51, 62, 62, 51, 16, -52], [47, 51, 21, -33, -33, 21, 51, 47], [39, 62, -33, -53, -53, -33, 62, 39], [39, 62, -33, -53, -53, -33, 62, 39], [47, 51, 21, -33, -33, 21, 51, 47], [-52, 16, 51, 62, 62, 51, 16, -52], [58, -52, 47, 39, 39, 47, -52, 58]]
                                  )
        self.weight_3 = np.array(
            [[48, -20, 31, 38, 38, 31, -20, 48], [-20, 19, -34, 29, 29, -34, 19, -20], [31, -34, 47, 23, 23, 47, -34, 31], [38, 29, 23, -10, -10, 23, 29, 38], [38, 29, 23, -10, -10, 23, 29, 38], [31, -34, 47, 23, 23, 47, -34, 31], [-20, 19, -34, 29, 29, -34, 19, -20], [48, -20, 31, 38, 38, 31, -20, 48]]
        )
        self.mobility_1_2 = np.array(
            [[-5, 34, 6, -13, -13, 6, 34, -5], [34, -29, -4, 7, 7, -4, -29, 34], [6, -4, 16, 29, 29, 16, -4, 6], [-13, 7, 29, 6, 6, 29, 7, -13], [-13, 7, 29, 6, 6, 29, 7, -13], [6, -4, 16, 29, 29, 16, -4, 6], [34, -29, -4, 7, 7, -4, -29, 34], [-5, 34, 6, -13, -13, 6, 34, -5]]
        )
        self.mobility_3 = np.array(
            [[-41, 63, 16, 48, 48, 16, 63, -41], [63, 56, -11, 12, 12, -11, 56, 63], [16, -11, 8, 6, 6, 8, -11, 16], [48, 12, 6, 58, 58, 6, 12, 48], [48, 12, 6, 58, 58, 6, 12, 48], [16, -11, 8, 6, 6, 8, -11, 16], [63, 56, -11, 12, 12, -11, 56, 63], [-41, 63, 16, 48, 48, 16, 63, -41]]
                                   )

    # The input is the current chessboard. Chessboard is a numpy array
    def go(self, chessboard):
        self.start_time = time.time()
        self.candidate_list.clear()
        self.candidate_list = self.get_valid_places(chessboard, self.color)
        if len(self.candidate_list) == 0:
            return self.candidate_list
        self.max_depth = FIRST_MAX_DEPTH
        self.depth_1_state = 1
        action_dep_1 = self.alpha_beta(chessboard, 1, self.color, float("-inf"), float("inf"))[1]
        self.depth_1_state = 2
        self.depth_3_state = 1
        self.depth_4_state = 0
        # 0：未开始  1：正常开始未结束  2：已正常结束  3：中途被打断  4：异常开始未结束
        self.max_depth = SECOND_MAX_DEPTH
        action_dep_3 = self.alpha_beta(chessboard, 1, self.color, float("-inf"), float("inf"))[1]
        if self.depth_3_state != 3:
            self.depth_3_state = 2
        self.max_depth = THIRD_MAX_DEPTH
        if self.depth_3_state != 3:
            self.depth_4_state = 1
        else:
            self.depth_4_state = 4
        action_dep_4 = self.alpha_beta(chessboard, 1, self.color, float("-inf"), float("inf"))[1]
        if self.depth_4_state != 3 and self.depth_4_state != 4:
            self.depth_4_state = 2
        if self.depth_3_state == 3:
            self.candidate_list.append(action_dep_1)
            print(FIRST_MAX_DEPTH)
            # print(FIRST_MAX_DEPTH)
        elif self.depth_4_state == 3:
            self.candidate_list.append(action_dep_3)
            print(SECOND_MAX_DEPTH)
            # print(SECOND_MAX_DEPTH)
        else:
            self.candidate_list.append(action_dep_4)
            print(THIRD_MAX_DEPTH)
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
                    continue
                if chessboard[x_][y_] == 0:
                    break
                if chessboard[x_][y_] == color:
                    for i, j in passed:
                        chessboard[i][j] = color
                    break

        steps = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        x, y = pos
        chessboard[x][y] = color
        for dx, dy in steps:
            update_chessboard_line(chessboard, x, y, dx, dy, color)

    def max_value(self, chessboard, depth, color, a, b):
        if time.time() - self.start_time > self.time_out - TIME_BEFORE_OUT:
            if self.depth_3_state == 1:
                self.depth_3_state = 3
            if self.depth_4_state == 1:
                self.depth_4_state = 3
            return None, None

        if depth > self.max_depth:
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
            self.update_chessboard(next_chessboard, (x, y), color)
            grades2, _ = self.min_value(next_chessboard, depth, next_color, a, b)
            if self.depth_3_state == 3 or self.depth_4_state == 3:
                return None, None
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
            if self.depth_3_state == 1:
                self.depth_3_state = 3
            if self.depth_4_state == 1:
                self.depth_4_state = 3
            return None, None
        if depth > self.max_depth:
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
            self.update_chessboard(next_chessboard, (x, y), color)
            grades2, _ = self.max_value(next_chessboard, depth, next_color, a, b)
            if self.depth_3_state == 3 or self.depth_4_state == 3:
                return None, None
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
        c2 = [None, 215, 127, 238]
        c3 = [None,14, -41, -53]
        c4 = [None, -28, -44, -43]
        result = c1 * location_weight + \
                 c2[state] * stable + \
                 c3[state] * diff_num + \
                 c4[state] * front + \
                 mobility
        return result
